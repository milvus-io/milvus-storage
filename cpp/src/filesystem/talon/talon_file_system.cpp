// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/talon/talon_file_system.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/status.h>
#include <arrow/util/future.h>
#include <arrow/util/key_value_metadata.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "milvus-storage/common/log.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "talon/talon_bridge.h"

namespace milvus_storage::talon {
namespace {

constexpr std::size_t kObjectStatCacheCapacity = 4096;

extern "C" void talon_dispatch(void (*callback)(void*), void* context);

using TalonTask = std::function<void()>;

struct TalonFileSystemState {
  TalonFileSystemState(ArrowFileSystemConfig config,
                       std::shared_ptr<TalonClient> client,
                       const arrow::io::IOContext& io_context)
      : config(std::move(config)), client(std::move(client)), io_context(io_context) {}

  const ArrowFileSystemConfig config;
  const std::shared_ptr<TalonClient> client;
  const arrow::io::IOContext io_context;
  // Shared by open files so pending initialization can fill the bounded cache
  // even after the filesystem wrapper is released. Objects remain TTL-free.
  LRUCache<std::string, TalonObjectStat> object_stats{kObjectStatCacheCapacity};
};

class TalonInputFile final : public arrow::io::RandomAccessFile, public NonBlockingRandomAccessFile {
  using Reader = std::shared_ptr<TalonObjectReader>;
  using Metadata = std::shared_ptr<const arrow::KeyValueMetadata>;

  public:
  TalonInputFile(std::shared_ptr<TalonFileSystemState> filesystem,
                 std::shared_ptr<arrow::io::RandomAccessFile> origin_file,
                 std::string key,
                 Reader reader)
      : state_(std::make_shared<ReaderState>(
            std::move(filesystem), std::move(origin_file), std::move(key), std::move(reader))),
        pool_(state_->filesystem->io_context.pool()) {}

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    return state_->origin_file->ReadMetadata();
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    if (closed_) {
      return arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(
          arrow::Status::Invalid("Operation on closed Talon input file"));
    }
    return state_->origin_file->ReadMetadataAsync(io_context);
  }

  arrow::Result<int64_t> GetSize() override { return GetSizeAsync().result(); }

  arrow::Future<int64_t> GetSizeAsync() override {
    return EnsureReaderAsync().Then([](const Reader& reader) { return reader->KnownSize(); });
  }

  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    auto read_size = GetReadSize(position, nbytes);
    if (!read_size.ok()) {
      return arrow::Future<int64_t>::MakeFinished(read_size.status());
    }
    nbytes = read_size.ValueOrDie();
    if (nbytes > 0 && out == nullptr) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Talon read destination is null")));
    }

    if (nbytes == 0) {
      return arrow::Future<int64_t>::MakeFinished(0);
    }

    return EnsureReaderAsync().Then([position, nbytes, out, origin_file = state_->origin_file.get()](
                                        const Reader& reader) -> arrow::Future<int64_t> {
      auto read_size = GetReadSize(position, nbytes, reader->KnownSize());
      if (!read_size.ok()) {
        return arrow::Future<int64_t>::MakeFinished(read_size.status());
      }
      const auto length = read_size.ValueOrDie();
      if (length == 0) {
        return arrow::Future<int64_t>::MakeFinished(0);
      }
      // A failed Talon read can have written a prefix. Retry the entire range.
      return WithOriginFallback(
          reader->ReadAtAsync(static_cast<uint64_t>(position), static_cast<uint64_t>(length), out), origin_file,
          [position, length, out](NonBlockingRandomAccessFile& file) {
            return file.ReadAtAsyncInto(position, length, out);
          },
          [position, length, out](arrow::io::RandomAccessFile& file) { return file.ReadAt(position, length, out); });
    });
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return ReadAtAsyncInto(position, nbytes, reinterpret_cast<uint8_t*>(out)).result();
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes));
    // A lazy open may not have a size yet. The synchronous API can wait for
    // initialization before clamping and allocating on the calling thread.
    ARROW_ASSIGN_OR_RAISE(const auto reader, EnsureReaderAsync().result());
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes, reader->KnownSize()));
    ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes, pool_));
    ARROW_ASSIGN_OR_RAISE(const int64_t bytes_read, ReadAt(position, nbytes, buffer->mutable_data()));
    ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
    return std::shared_ptr<arrow::Buffer>(std::move(buffer));
  }

  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    auto read_size = GetReadSize(position, nbytes);
    if (!read_size.ok()) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(read_size.status());
    }
    nbytes = read_size.ValueOrDie();
    if (nbytes == 0) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(std::make_shared<arrow::Buffer>(nullptr, 0));
    }
    return EnsureReaderAsync().Then([position, nbytes, pool = io_context.pool(),
                                     origin_file = state_->origin_file.get()](
                                        const Reader& reader) -> arrow::Future<std::shared_ptr<arrow::Buffer>> {
      // A cold open has no size yet. Resolve and clamp before allocating,
      // including when the caller supplies a large read extending past EOF.
      auto read_size = GetReadSize(position, nbytes, reader->KnownSize());
      if (!read_size.ok()) {
        return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(read_size.status());
      }
      const auto length = read_size.ValueOrDie();
      auto maybe_buffer = arrow::AllocateResizableBuffer(length, pool);
      if (!maybe_buffer.ok()) {
        return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(maybe_buffer.status());
      }
      auto buffer = std::move(maybe_buffer).ValueOrDie();
      if (length == 0) {
        return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
            std::shared_ptr<arrow::Buffer>(std::move(buffer)));
      }
      auto* const out = buffer->mutable_data();
      return WithOriginFallback(
                 reader->ReadAtAsync(static_cast<uint64_t>(position), static_cast<uint64_t>(length), out), origin_file,
                 [position, length, out](NonBlockingRandomAccessFile& file) {
                   return file.ReadAtAsyncInto(position, length, out);
                 },
                 [position, length, out](arrow::io::RandomAccessFile& file) {
                   return file.ReadAt(position, length, out);
                 })
          .Then([buffer = std::move(buffer),
                 length](const int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
            if (bytes_read > length) {
              return arrow::Status::IOError("Talon returned more bytes than requested");
            }
            ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
            return std::shared_ptr<arrow::Buffer>(std::move(buffer));
          });
    });
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(const int64_t bytes_read, ReadAt(pos_, nbytes, out));
    pos_ += bytes_read;
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
    pos_ += buffer->size();
    return buffer;
  }

  arrow::Status Seek(int64_t position) override {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot seek to negative position");
    }
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    const int64_t known_size = state_->known_size.load(std::memory_order_acquire);
    if (known_size >= 0 && position > known_size) {
      return arrow::Status::IOError("Cannot seek past end of Talon input file");
    }
    pos_ = position;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    return pos_;
  }

  // Matches the lifecycle contract used by Arrow's S3 and Azure input files:
  // Close() is not safe to call concurrently with another operation on this
  // file. Pending size initialization retains its origin file until completion.
  // Data reads must keep this file open and alive until their futures finish:
  // a failed Talon read may still need to retry through the origin file.
  arrow::Status Close() override {
    if (closed_) {
      return arrow::Status::OK();
    }
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (!state_->reader.is_valid() || state_->reader.is_finished()) {
        ARROW_RETURN_NOT_OK(state_->origin_file->Close());
      }
    }
    state_.reset();
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  private:
  /// Retry a failed Talon operation once through the cached origin file.
  /// @ReadOriginAsync takes NonBlockingRandomAccessFile& and returns Future<int64_t>.
  /// @ReadOrigin takes arrow::io::RandomAccessFile& and returns Result<int64_t>.
  /// Use the async callback when supported, otherwise invoke the blocking callback.
  /// Both callbacks repeat the same operation; reads must retry the full range.
  /// Keep the file open and destination buffer valid until the returned future completes.
  template <typename ReadOriginAsync, typename ReadOrigin>
  static arrow::Future<int64_t> WithOriginFallback(arrow::Future<int64_t> future,
                                                   arrow::io::RandomAccessFile* const origin_file,
                                                   ReadOriginAsync read_origin_async,
                                                   ReadOrigin read_origin) {
    return future.Then(
        [](const int64_t value) { return value; },
        [origin_file, read_origin_async = std::move(read_origin_async),
         read_origin = std::move(read_origin)](const arrow::Status& talon_error) -> arrow::Future<int64_t> {
#if 0
          // TODO(jiaqizho): Replace fallback logging with metrics.
          LOG_STORAGE_WARNING_ << "Talon failed, falling back to origin filesystem: " << talon_error;
#endif
          arrow::Future<int64_t> origin_future;
          if (auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(origin_file)) {
            origin_future = read_origin_async(*async_file);
          } else {
            // Origins without native async support still work through their blocking API.
            origin_future = arrow::Future<int64_t>::MakeFinished(read_origin(*origin_file));
          }
          // Native CRT requests own their I/O state. Keep origin owners out of
          // their completion callbacks to avoid destroying a CRT client there.
          return origin_future.Then([](const int64_t value) { return value; },
                                    [talon_error](const arrow::Status& origin_error) -> arrow::Result<int64_t> {
                                      // Preserve the provider's status code and detail for caller classification.
                                      return origin_error.WithMessage(
                                          "Talon failed: ", talon_error.ToString(),
                                          "; origin filesystem fallback failed: ", origin_error.message());
                                    });
        });
  }

  arrow::Result<int64_t> GetReadSize(const int64_t position, const int64_t nbytes) const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    return GetReadSize(position, nbytes, state_->known_size.load(std::memory_order_acquire));
  }

  static arrow::Result<int64_t> GetReadSize(const int64_t position, const int64_t nbytes, const int64_t known_size) {
    if (position < 0) {
      return arrow::Status::Invalid("Cannot read from negative position");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Cannot read a negative number of bytes");
    }
    // Match S3's EOF handling before allocating a buffer. Only consult cached
    // metadata so an unknown size never introduces a blocking stat here.
    if (known_size >= 0) {
      if (position > known_size) {
        return arrow::Status::IOError("Cannot read past end of file");
      }
      return std::min(nbytes, known_size - position);
    }
    return nbytes;
  }

  struct ReaderState {
    ReaderState(std::shared_ptr<TalonFileSystemState> filesystem,
                std::shared_ptr<arrow::io::RandomAccessFile> origin_file,
                std::string key,
                Reader initial_reader)
        : filesystem(std::move(filesystem)),
          origin_file(std::move(origin_file)),
          key(std::move(key)),
          known_size(initial_reader == nullptr ? -1 : initial_reader->KnownSize()),
          reader(initial_reader == nullptr ? arrow::Future<Reader>()
                                           : arrow::Future<Reader>::MakeFinished(std::move(initial_reader))) {}

    ~ReaderState() {
      if (!origin_file->closed()) {
        (void)origin_file->Close();
      }
    }

    const std::shared_ptr<TalonFileSystemState> filesystem;
    const std::shared_ptr<arrow::io::RandomAccessFile> origin_file;
    const std::string key;
    std::atomic<int64_t> known_size;
    std::mutex mutex;
    arrow::Future<Reader> reader;
    arrow::Result<Metadata> metadata;
  };

  static void FinishInitialization(std::shared_ptr<ReaderState> state) noexcept {
    // Completion can synchronously trigger a retry that replaces state->reader.
    // Keep this attempt's Future alive independently of the state member.
    auto ready = state->reader;
    auto metadata_result = std::move(state->metadata);
    arrow::Result<Reader> result;
    try {
      result = [&]() -> arrow::Result<Reader> {
        ARROW_ASSIGN_OR_RAISE(const auto metadata, metadata_result);
        const auto& filesystem = *state->filesystem;
        const std::string path = filesystem.config.bucket_name + "/" + state->key;
        if (metadata == nullptr || !metadata->Contains("ETag")) {
          return arrow::Status::IOError("Missing ETag metadata for Talon object: ", path);
        }
        ARROW_ASSIGN_OR_RAISE(auto version, metadata->Get("ETag"));
        // Talon's versioned cache and If-Match use the unquoted origin ETag.
        boost::trim_if(version, boost::is_any_of("\""));
        if (boost::all(version, boost::is_any_of(" \t\r\n"))) {
          return arrow::Status::IOError("Empty ETag metadata for Talon object: ", path);
        }
        // The provider caches its size with metadata in the same HEAD.
        ARROW_ASSIGN_OR_RAISE(const auto size, state->origin_file->GetSize());
        if (size < 0) {
          return arrow::Status::IOError("Invalid size for Talon object: ", path, ": ", size);
        }
        const TalonObjectStat stat{static_cast<uint64_t>(size), std::move(version)};
        ARROW_ASSIGN_OR_RAISE(
            auto reader, filesystem.client->OpenObject(filesystem.config.cloud_provider, filesystem.config.bucket_name,
                                                       state->key, stat));
        auto shared_reader = std::make_shared<TalonObjectReader>(std::move(reader));
        state->filesystem->object_stats.put(state->key, stat);
        state->known_size.store(size, std::memory_order_release);
        return shared_reader;
      }();
    } catch (const std::exception& error) {
      result = arrow::Status::IOError("Failed to initialize Talon reader: ", error.what());
    }
    // Never complete under the state lock: continuations can retry an error
    // or submit another read immediately. The result owns no ReaderState.
    ready.MarkFinished(std::move(result));
  }

  static void StartInitialization(std::shared_ptr<ReaderState> state) noexcept {
    try {
      // Prepare the owning task before HEAD completion. The provider callback
      // only borrows its state, then transfers the whole task to the runtime.
      auto finish = std::make_unique<TalonTask>([state] { FinishInitialization(state); });
      auto metadata = state->origin_file->ReadMetadataAsync(state->filesystem->io_context);
      metadata.AddCallback(
          [state = state.get(), finish = std::move(finish)](const arrow::Result<Metadata>& result) mutable {
            state->metadata = result;
            // Reclaim the task and its captured state on the runtime thread.
            talon_dispatch(
                [](void* context) {
                  std::unique_ptr<TalonTask> task(static_cast<TalonTask*>(context));
                  (*task)();
                },
                finish.release());
          });
    } catch (const std::exception& error) {
      state->metadata = arrow::Status::IOError("Failed to read Talon metadata: ", error.what());
      FinishInitialization(std::move(state));
    }
  }

  arrow::Future<Reader> EnsureReaderAsync() {
    if (closed_) {
      return arrow::Future<Reader>::MakeFinished(arrow::Status::Invalid("Operation on closed Talon input file"));
    }
    arrow::Future<Reader> ready;
    std::unique_ptr<TalonTask> start;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (state_->reader.is_valid() && (!state_->reader.is_finished() || state_->reader.result().ok())) {
        return state_->reader;
      }
      ready = arrow::Future<Reader>::Make();
      start = std::make_unique<TalonTask>([state = state_] { StartInitialization(state); });
      state_->reader = ready;
    }
    // This first dispatch runs on the submitting thread, before any HEAD, so
    // lazy Tokio runtime construction can never happen on a CRT callback.
    talon_dispatch(
        [](void* context) {
          std::unique_ptr<TalonTask> task(static_cast<TalonTask*>(context));
          (*task)();
        },
        start.release());
    return ready;
  }

  std::shared_ptr<ReaderState> state_;
  arrow::MemoryPool* const pool_;
  int64_t pos_ = 0;
  bool closed_ = false;
};

class TalonFileSystem final : public arrow::fs::FileSystem,
                              public UploadConditional,
                              public UploadSizable,
                              public Observable {
  public:
  TalonFileSystem(ArrowFileSystemConfig config, ArrowFileSystemPtr origin_fs, std::shared_ptr<TalonClient> client)
      : arrow::fs::FileSystem(origin_fs->io_context()),
        origin_fs_(std::move(origin_fs)),
        state_(std::make_shared<TalonFileSystemState>(std::move(config), std::move(client), io_context())) {}

  std::string type_name() const override { return origin_fs_->type_name(); }

  arrow::Result<std::string> NormalizePath(std::string path) override {
    return origin_fs_->NormalizePath(std::move(path));
  }

  arrow::Result<std::string> PathFromUri(const std::string& uri) const override { return origin_fs_->PathFromUri(uri); }

  arrow::Result<std::string> MakeUri(std::string path) const override { return origin_fs_->MakeUri(std::move(path)); }

  bool Equals(const arrow::fs::FileSystem& other) const override {
    if (this == &other) {
      return true;
    }
    const auto* talon = dynamic_cast<const TalonFileSystem*>(&other);
    return talon != nullptr && state_->config.bucket_name == talon->state_->config.bucket_name &&
           state_->config.cloud_provider == talon->state_->config.cloud_provider &&
           state_->config.talon_coordinator == talon->state_->config.talon_coordinator &&
           state_->config.talon_block_size == talon->state_->config.talon_block_size &&
           state_->config.talon_max_idle_per_addr == talon->state_->config.talon_max_idle_per_addr &&
           origin_fs_->Equals(*talon->origin_fs_);
  }

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    return origin_fs_->GetFileInfo(path);
  }

  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& selector) override {
    return origin_fs_->GetFileInfo(selector);
  }

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) override {
    return origin_fs_->GetFileInfoGenerator(selector);
  }

  arrow::Status CreateDir(const std::string& path, bool recursive) override {
    return origin_fs_->CreateDir(path, recursive);
  }

  arrow::Status DeleteDir(const std::string& path) override { return origin_fs_->DeleteDir(path); }

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
    return origin_fs_->DeleteDirContents(path, missing_dir_ok);
  }

  arrow::Status DeleteRootDirContents() override { return origin_fs_->DeleteRootDirContents(); }

  arrow::Status DeleteFile(const std::string& path) override { return origin_fs_->DeleteFile(path); }

  arrow::Status Move(const std::string& src, const std::string& dest) override { return origin_fs_->Move(src, dest); }

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override {
    return origin_fs_->CopyFile(src, dest);
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(path));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    ARROW_ASSIGN_OR_RAISE(auto file, OpenInputFile(info));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    return OpenTalon(path);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    // Reject a known non-file before resolving its object metadata.
    if (info.type() == arrow::fs::FileType::NotFound) {
      return arrow::fs::internal::PathNotFound(info.path());
    }
    if (info.type() != arrow::fs::FileType::File && info.type() != arrow::fs::FileType::Unknown) {
      return arrow::fs::internal::NotAFile(info.path());
    }
    return OpenTalon(info.path());
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    return origin_fs_->OpenOutputStream(path, metadata);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    return origin_fs_->OpenAppendStream(path, metadata);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) override {
    const auto conditional = std::dynamic_pointer_cast<UploadConditional>(origin_fs_);
    if (conditional == nullptr) {
      return arrow::Status::NotImplemented("Talon cannot forward conditional output stream for path '", path,
                                           "': origin filesystem type '", origin_fs_->type_name(),
                                           "' does not implement UploadConditional");
    }
    return conditional->OpenConditionalOutputStream(path, std::move(metadata));
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& path,
      const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
      int64_t part_size) override {
    const auto sizable = std::dynamic_pointer_cast<UploadSizable>(origin_fs_);
    if (sizable == nullptr) {
      return arrow::Status::NotImplemented("Talon cannot forward sized output stream for path '", path,
                                           "' with part size ", part_size, ": origin filesystem type '",
                                           origin_fs_->type_name(), "' does not implement UploadSizable");
    }
    return sizable->OpenOutputStreamWithUploadSize(path, metadata, part_size);
  }

  std::shared_ptr<FilesystemMetrics> GetMetrics() const override {
    // TODO(jiaqizho): Add dedicated Talon metrics to distinguish accesses through Talon from
    // direct cloud-provider accesses, keeping them separate from these origin metrics.
    const auto observable = std::dynamic_pointer_cast<Observable>(origin_fs_);
    return observable == nullptr ? nullptr : observable->GetMetrics();
  }

  private:
  arrow::Result<std::string> ObjectKey(const std::string& path) const {
    const std::string prefix = state_->config.bucket_name + "/";
    if (!path.starts_with(prefix)) {
      return arrow::Status::Invalid("Talon path does not belong to configured bucket ", state_->config.bucket_name,
                                    ": ", path);
    }
    const std::string key = path.substr(prefix.size());
    if (key.empty()) {
      return arrow::Status::Invalid("Talon object key must be non-empty");
    }
    return key;
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalon(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(const auto key, ObjectKey(path));
    ARROW_ASSIGN_OR_RAISE(auto origin_file, origin_fs_->OpenInputFile(path));
    std::shared_ptr<TalonObjectReader> reader;
    if (const auto initial_stat = state_->object_stats.get(key); initial_stat.has_value()) {
      ARROW_ASSIGN_OR_RAISE(
          auto cached_reader,
          state_->client->OpenObject(state_->config.cloud_provider, state_->config.bucket_name, key, initial_stat));
      reader = std::make_shared<TalonObjectReader>(std::move(cached_reader));
    }
    return std::make_shared<TalonInputFile>(state_, std::move(origin_file), key, std::move(reader));
  }

  const ArrowFileSystemPtr origin_fs_;
  const std::shared_ptr<TalonFileSystemState> state_;
};

}  // namespace

namespace internal {

arrow::Result<ArrowFileSystemPtr> MakeTalonFileSystem(const ArrowFileSystemConfig& config,
                                                      ArrowFileSystemPtr origin_fs,
                                                      std::string bucket) {
  ARROW_ASSIGN_OR_RAISE(auto client, TalonClient::Make(config.talon_coordinator, config.talon_block_size,
                                                       config.talon_max_idle_per_addr));
  auto talon_config = config;
  // Readers use the producer's normalized bucket rather than the original spelling.
  talon_config.bucket_name = std::move(bucket);
  return std::make_shared<TalonFileSystem>(std::move(talon_config), std::move(origin_fs), std::move(client));
}

}  // namespace internal

}  // namespace milvus_storage::talon
