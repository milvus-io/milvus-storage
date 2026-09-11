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
#include <cstdint>
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

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "talon/talon_bridge.h"

namespace milvus_storage::talon {
namespace {

class TalonInputFile final : public arrow::io::RandomAccessFile, public NonBlockingRandomAccessFile {
  public:
  TalonInputFile(TalonObjectReader reader,
                 ArrowFileSystemPtr origin_fs,
                 std::string path,
                 arrow::MemoryPool* const pool)
      : reader_(std::move(reader)), origin_fs_(std::move(origin_fs)), path_(std::move(path)), pool_(pool) {}

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    ARROW_ASSIGN_OR_RAISE(auto file, GetOrOpenOriginFile());
    return file->ReadMetadata();
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    auto file = GetOrOpenOriginFile();
    if (!file.ok()) {
      return arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(file.status());
    }
    return file.ValueOrDie()->ReadMetadataAsync(io_context);
  }

  arrow::Result<int64_t> GetSize() override { return GetSizeAsync().result(); }

  arrow::Future<int64_t> GetSizeAsync() override {
    if (closed_) {
      return arrow::Future<int64_t>::MakeFinished(
          arrow::Result<int64_t>(arrow::Status::Invalid("Operation on closed Talon input file")));
    }
    const int64_t known_size = reader_->KnownSize();
    if (known_size >= 0) {
      return arrow::Future<int64_t>::MakeFinished(known_size);
    }

    return reader_->StatAsync();
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

    return reader_->ReadAtAsync(static_cast<uint64_t>(position), static_cast<uint64_t>(nbytes), out);
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return ReadAtAsyncInto(position, nbytes, reinterpret_cast<uint8_t*>(out)).result();
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(nbytes, GetReadSize(position, nbytes));
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
    auto maybe_buffer = arrow::AllocateResizableBuffer(nbytes, io_context.pool());
    if (!maybe_buffer.ok()) {
      return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
          arrow::Result<std::shared_ptr<arrow::Buffer>>(maybe_buffer.status()));
    }
    auto buffer = std::move(maybe_buffer).ValueOrDie();
    auto* const out = buffer->mutable_data();
    return ReadAtAsyncInto(position, nbytes, out)
        .Then([buffer = std::move(buffer),
               nbytes](const int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
          if (bytes_read > nbytes) {
            return arrow::Status::IOError("Talon returned more bytes than requested");
          }
          ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
          return std::shared_ptr<arrow::Buffer>(std::move(buffer));
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
    const int64_t known_size = reader_->KnownSize();
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
  // file. Reads whose submission completed before Close() was called own a
  // Rust reader clone and may complete after the C++ handle is released here.
  arrow::Status Close() override {
    if (origin_file_ != nullptr) {
      ARROW_RETURN_NOT_OK(origin_file_->Close());
      origin_file_.reset();
    }
    origin_fs_.reset();
    reader_.reset();
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override { return closed_; }

  private:
  arrow::Result<int64_t> GetReadSize(const int64_t position, const int64_t nbytes) const {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    if (position < 0) {
      return arrow::Status::Invalid("Cannot read from negative position");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Cannot read a negative number of bytes");
    }
    // Match S3's EOF handling before allocating a buffer. Only consult cached
    // metadata so an unknown size never introduces a blocking stat here.
    const int64_t known_size = reader_->KnownSize();
    if (known_size >= 0) {
      if (position > known_size) {
        return arrow::Status::IOError("Cannot read past end of file");
      }
      return std::min(nbytes, known_size - position);
    }
    return nbytes;
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> GetOrOpenOriginFile() {
    if (closed_) {
      return arrow::Status::Invalid("Operation on closed Talon input file");
    }
    // Share the provider client and open a file only when metadata is requested.
    // Keep the successfully opened file until Close(), preserving the provider's
    // metadata cache without adding origin opens or HEADs to ordinary Talon reads.
    const std::lock_guard<std::mutex> lock(origin_file_mutex_);
    if (origin_file_ == nullptr) {
      ARROW_ASSIGN_OR_RAISE(origin_file_, origin_fs_->OpenInputFile(path_));
    }
    return origin_file_;
  }

  std::optional<TalonObjectReader> reader_;
  ArrowFileSystemPtr origin_fs_;
  const std::string path_;
  std::mutex origin_file_mutex_;
  std::shared_ptr<arrow::io::RandomAccessFile> origin_file_;
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
        client_(std::move(client)),
        config_(std::move(config)) {}

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
    return talon != nullptr && config_.bucket_name == talon->config_.bucket_name &&
           config_.cloud_provider == talon->config_.cloud_provider &&
           config_.talon_coordinator == talon->config_.talon_coordinator &&
           config_.talon_block_size == talon->config_.talon_block_size &&
           config_.talon_max_idle_per_addr == talon->config_.talon_max_idle_per_addr &&
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
    ARROW_ASSIGN_OR_RAISE(auto file, OpenTalon(path, arrow::fs::kNoSize));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    ARROW_ASSIGN_OR_RAISE(auto file, OpenInputFile(info));
    return std::static_pointer_cast<arrow::io::InputStream>(std::move(file));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    return OpenTalon(path, arrow::fs::kNoSize);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    // Honor the caller's known type without issuing another metadata request.
    if (info.type() == arrow::fs::FileType::NotFound) {
      return arrow::fs::internal::PathNotFound(info.path());
    }
    if (info.type() != arrow::fs::FileType::File && info.type() != arrow::fs::FileType::Unknown) {
      return arrow::fs::internal::NotAFile(info.path());
    }
    return OpenTalon(info.path(), info.size());
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
    const std::string prefix = config_.bucket_name + "/";
    if (!path.starts_with(prefix)) {
      return arrow::Status::Invalid("Talon path does not belong to configured bucket ", config_.bucket_name, ": ",
                                    path);
    }
    const std::string key = path.substr(prefix.size());
    if (key.empty()) {
      return arrow::Status::Invalid("Talon object key must be non-empty");
    }
    return key;
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalon(const std::string& path,
                                                                        int64_t known_size) const {
    ARROW_ASSIGN_OR_RAISE(auto key, ObjectKey(path));
    if (known_size < arrow::fs::kNoSize) {
      return arrow::Status::Invalid("Invalid known Talon object size: ", known_size);
    }
    const auto initial_stat =
        known_size == arrow::fs::kNoSize
            ? std::nullopt
            : std::optional<TalonObjectStat>{TalonObjectStat{static_cast<uint64_t>(known_size), ""}};
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          client_->OpenObject(config_.cloud_provider, config_.bucket_name, key, initial_stat));
    return std::make_shared<TalonInputFile>(std::move(reader), origin_fs_, path, io_context().pool());
  }

  const ArrowFileSystemPtr origin_fs_;
  const std::shared_ptr<TalonClient> client_;
  const ArrowFileSystemConfig config_;
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
