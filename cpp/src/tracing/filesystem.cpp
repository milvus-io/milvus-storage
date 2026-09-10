// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "tracing/filesystem.h"
#include <arrow/buffer.h>
#include <limits>
#include "tracing/runtime.h"
#include "milvus-storage/filesystem/async_random_access_file.h"

namespace milvus_storage::tracing {
namespace {
int64_t Bytes(int64_t value) { return value; }
int64_t Bytes(const std::shared_ptr<arrow::Buffer>& buffer) { return buffer ? buffer->size() : 0; }

class TracedFile : public arrow::io::RandomAccessFile {
  public:
  TracedFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::string backend)
      : file_(std::move(file)), backend_(std::move(backend)) {}
  arrow::Status Close() override { return file_->Close(); }
  arrow::Future<> CloseAsync() override { return file_->CloseAsync(); }
  arrow::Status Abort() override { return file_->Abort(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }
  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }
  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return Run("storage.fs.metadata", [&] { return file_->ReadMetadata(); }, true);
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& context) override {
    OperationTrace trace("storage.fs.metadata", false, true);
    ContextScope scope(trace.context());
    try {
      return Observe(file_->ReadMetadataAsync(context), trace);
    } catch (...) {
      auto status = arrow::Status::UnknownError("exception");
      trace.Finish(status);
      return arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(status);
    }
  }
  arrow::Result<int64_t> GetSize() override {
    return Run("storage.fs.head", [&] { return file_->GetSize(); }, true);
  }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    return ReadSync(-1, nbytes, [&] { return file_->Read(nbytes, out); });
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    return ReadSync(-1, nbytes, [&] { return file_->Read(nbytes); });
  }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return ReadSync(position, nbytes, [&] { return file_->ReadAt(position, nbytes, out); });
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    return ReadSync(position, nbytes, [&] { return file_->ReadAt(position, nbytes); });
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    return ReadFuture(position, nbytes, [&] { return file_->ReadAsync(context, position, nbytes); });
  }
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
      const arrow::io::IOContext& context, const std::vector<arrow::io::ReadRange>& ranges) override {
    if (!HasContext())
      return file_->ReadManyAsync(context, ranges);
    OperationTrace trace("storage.fs.read", false, true);
    ContextScope scope(trace.context());
    try {
      if (!trace.IsEnabled())
        return file_->ReadManyAsync(context, ranges);
      trace.Attribute("storage.backend", backend_.c_str());
      trace.Attribute("storage.range_count", static_cast<int64_t>(ranges.size()));
      int64_t requested = 0;
      for (const auto& range : ranges) {
        auto length = std::max<int64_t>(0, range.length);
        requested += std::min(length, std::numeric_limits<int64_t>::max() - requested);
      }
      trace.Attribute("storage.requested_bytes", requested);
      struct Completion {
        explicit Completion(size_t count, OperationTrace trace) : remaining(count), trace(std::move(trace)) {}
        std::mutex mutex;
        size_t remaining;
        int64_t bytes = 0;
        arrow::Status status;
        OperationTrace trace;
      };
      auto futures = file_->ReadManyAsync(context, ranges);
      if (futures.empty())
        trace.Finish(arrow::Status::OK());
      auto completion = std::make_shared<Completion>(futures.size(), trace);
      for (auto& future : futures) {
        future.AddCallback([completion, requested](const arrow::Result<std::shared_ptr<arrow::Buffer>>& result) {
          std::lock_guard<std::mutex> lock(completion->mutex);
          if (!result.ok())
            completion->status = result.status();
          else
            completion->bytes += std::min(Bytes(*result), std::numeric_limits<int64_t>::max() - completion->bytes);
          if (--completion->remaining == 0) {
            completion->trace.AccountRead(requested, completion->bytes);
            completion->trace.Attribute("storage.returned_bytes", completion->bytes);
            completion->trace.Finish(completion->status);
          }
        });
      }
      return futures;
    } catch (...) {
      auto status = arrow::Status::UnknownError("exception");
      trace.Finish(status);
      return std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>>(
          ranges.size(), arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(status));
    }
  }

  protected:
  template <typename F>
  auto ReadSync(int64_t position, int64_t nbytes, F&& fn) -> decltype(fn()) {
    if (!HasContext())
      return fn();
    return ReadSyncTraced(position, nbytes, std::forward<F>(fn));
  }
  // Keep the recording path out of the common forwarding path's stack frame.
  template <typename F>
  FOLLY_NOINLINE auto ReadSyncTraced(int64_t position, int64_t nbytes, F&& fn) -> decltype(fn()) {
    OperationTrace trace("storage.fs.read", false, true);
    ContextScope scope(trace.context());
    Attributes(trace, position, nbytes);
    try {
      auto result = fn();
      trace.AccountRead(nbytes, result.ok() ? Bytes(*result) : 0);
      if (result.ok())
        trace.Attribute("storage.returned_bytes", Bytes(*result));
      trace.Finish(result.status());
      return result;
    } catch (...) {
      auto status = arrow::Status::UnknownError("exception");
      trace.Finish(status);
      return status;
    }
  }
  template <typename F>
  auto ReadFuture(int64_t position, int64_t nbytes, F&& fn) -> decltype(fn()) {
    if (!HasContext())
      return fn();
    return ReadFutureTraced(position, nbytes, std::forward<F>(fn));
  }
  template <typename F>
  FOLLY_NOINLINE auto ReadFutureTraced(int64_t position, int64_t nbytes, F&& fn) -> decltype(fn()) {
    OperationTrace trace("storage.fs.read", false, true);
    ContextScope scope(trace.context());
    Attributes(trace, position, nbytes);
    try {
      auto future = fn();
      if (!trace.IsEnabled())
        return future;
      future.AddCallback([trace, nbytes](const arrow::Result<typename decltype(future)::ValueType>& result) {
        trace.AccountRead(nbytes, result.ok() ? Bytes(*result) : 0);
        if (result.ok()) {
          trace.Attribute("storage.returned_bytes", Bytes(*result));
          trace.Attribute("storage.short_read", static_cast<int64_t>(Bytes(*result) < nbytes));
        }
        trace.Finish(result.status());
      });
      return future;
    } catch (...) {
      auto status = arrow::Status::UnknownError("exception");
      trace.Finish(status);
      return decltype(fn())::MakeFinished(status);
    }
  }
  void Attributes(const OperationTrace& trace, int64_t position, int64_t nbytes) {
    trace.Attribute("storage.backend", backend_.c_str());
    trace.Attribute("storage.offset", position);
    trace.Attribute("storage.requested_bytes", nbytes);
  }
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::string backend_;
};

// Only expose this capability when the underlying file implements it. Never
// fall back to blocking I/O from a NonBlockingRandomAccessFile method.
class TracedAsyncFile final : public TracedFile, public NonBlockingRandomAccessFile {
  public:
  TracedAsyncFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::string backend)
      : TracedFile(std::move(file), std::move(backend)),
        async_(dynamic_cast<NonBlockingRandomAccessFile*>(file_.get())) {}
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    return ReadFuture(position, nbytes, [&] { return async_->ReadAtAsyncInto(position, nbytes, out); });
  }
  arrow::Future<int64_t> GetSizeAsync() override {
    OperationTrace trace("storage.fs.head", false, true);
    ContextScope scope(trace.context());
    try {
      return Observe(async_->GetSizeAsync(), trace);
    } catch (...) {
      auto status = arrow::Status::UnknownError("exception");
      trace.Finish(status);
      return arrow::Future<int64_t>::MakeFinished(status);
    }
  }

  private:
  NonBlockingRandomAccessFile* async_;
};
}  // namespace
std::shared_ptr<arrow::io::RandomAccessFile> WrapFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                                      std::string backend) {
  if (dynamic_cast<TracedFile*>(file.get()))
    return file;
  if (dynamic_cast<NonBlockingRandomAccessFile*>(file.get()))
    return std::make_shared<TracedAsyncFile>(std::move(file), std::move(backend));
  return std::make_shared<TracedFile>(std::move(file), std::move(backend));
}
}  // namespace milvus_storage::tracing
