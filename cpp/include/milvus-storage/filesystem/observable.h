// Copyright 2024 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage {

/// \brief Metrics collection for filesystem operations
class FilesystemMetrics {
  public:
  FilesystemMetrics() = default;
  ~FilesystemMetrics() = default;

  // Counter increment methods
  void IncrementReadCount() { read_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementWriteCount() { write_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementReadBytes(int64_t bytes) { read_bytes_.fetch_add(bytes, std::memory_order_relaxed); }
  void IncrementWriteBytes(int64_t bytes) { write_bytes_.fetch_add(bytes, std::memory_order_relaxed); }
  void IncrementGetFileInfoCount() { get_file_info_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementCreateDirCount() { create_dir_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementDeleteDirCount() { delete_dir_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementDeleteFileCount() { delete_file_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementMoveCount() { move_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementCopyFileCount() { copy_file_count_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementFailedCount() { failed_count_.fetch_add(1, std::memory_order_relaxed); }

  // S3-specific metrics
  void IncrementMultiPartUploadCreated() { multi_part_upload_created_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementMultiPartUploadFinished() { multi_part_upload_finished_.fetch_add(1, std::memory_order_relaxed); }

  // Getter methods
  int64_t GetReadCount() const { return read_count_.load(std::memory_order_relaxed); }
  int64_t GetWriteCount() const { return write_count_.load(std::memory_order_relaxed); }
  int64_t GetReadBytes() const { return read_bytes_.load(std::memory_order_relaxed); }
  int64_t GetWriteBytes() const { return write_bytes_.load(std::memory_order_relaxed); }
  int64_t GetGetFileInfoCount() const { return get_file_info_count_.load(std::memory_order_relaxed); }
  int64_t GetCreateDirCount() const { return create_dir_count_.load(std::memory_order_relaxed); }
  int64_t GetDeleteDirCount() const { return delete_dir_count_.load(std::memory_order_relaxed); }
  int64_t GetDeleteFileCount() const { return delete_file_count_.load(std::memory_order_relaxed); }
  int64_t GetMoveCount() const { return move_count_.load(std::memory_order_relaxed); }
  int64_t GetCopyFileCount() const { return copy_file_count_.load(std::memory_order_relaxed); }
  int64_t GetFailedCount() const { return failed_count_.load(std::memory_order_relaxed); }

  // S3-specific getters
  int64_t GetMultiPartUploadCreated() const { return multi_part_upload_created_.load(std::memory_order_relaxed); }
  int64_t GetMultiPartUploadFinished() const { return multi_part_upload_finished_.load(std::memory_order_relaxed); }

  /// \brief Reset all metrics to zero
  void Reset() {
    read_count_.store(0, std::memory_order_relaxed);
    write_count_.store(0, std::memory_order_relaxed);
    read_bytes_.store(0, std::memory_order_relaxed);
    write_bytes_.store(0, std::memory_order_relaxed);
    get_file_info_count_.store(0, std::memory_order_relaxed);
    create_dir_count_.store(0, std::memory_order_relaxed);
    delete_dir_count_.store(0, std::memory_order_relaxed);
    delete_file_count_.store(0, std::memory_order_relaxed);
    move_count_.store(0, std::memory_order_relaxed);
    copy_file_count_.store(0, std::memory_order_relaxed);
    failed_count_.store(0, std::memory_order_relaxed);
    multi_part_upload_created_.store(0, std::memory_order_relaxed);
    multi_part_upload_finished_.store(0, std::memory_order_relaxed);
  }

  /// \brief Structured snapshot of all metrics
  struct MetricsSnapshot {
    int64_t read_count = 0;
    int64_t write_count = 0;
    int64_t read_bytes = 0;
    int64_t write_bytes = 0;
    int64_t get_file_info_count = 0;
    int64_t create_dir_count = 0;
    int64_t delete_dir_count = 0;
    int64_t delete_file_count = 0;
    int64_t move_count = 0;
    int64_t copy_file_count = 0;
    int64_t failed_count = 0;
    // S3-specific metrics
    int64_t multi_part_upload_created = 0;
    int64_t multi_part_upload_finished = 0;
  };

  MetricsSnapshot GetSnapshot() const {
    MetricsSnapshot snapshot;
    snapshot.read_count = GetReadCount();
    snapshot.write_count = GetWriteCount();
    snapshot.read_bytes = GetReadBytes();
    snapshot.write_bytes = GetWriteBytes();
    snapshot.get_file_info_count = GetGetFileInfoCount();
    snapshot.create_dir_count = GetCreateDirCount();
    snapshot.delete_dir_count = GetDeleteDirCount();
    snapshot.delete_file_count = GetDeleteFileCount();
    snapshot.move_count = GetMoveCount();
    snapshot.copy_file_count = GetCopyFileCount();
    snapshot.failed_count = GetFailedCount();
    snapshot.multi_part_upload_created = GetMultiPartUploadCreated();
    snapshot.multi_part_upload_finished = GetMultiPartUploadFinished();
    return snapshot;
  }

  private:
  std::atomic<int64_t> read_count_{0};
  std::atomic<int64_t> write_count_{0};
  std::atomic<int64_t> read_bytes_{0};
  std::atomic<int64_t> write_bytes_{0};
  std::atomic<int64_t> get_file_info_count_{0};
  std::atomic<int64_t> create_dir_count_{0};
  std::atomic<int64_t> delete_dir_count_{0};
  std::atomic<int64_t> delete_file_count_{0};
  std::atomic<int64_t> move_count_{0};
  std::atomic<int64_t> copy_file_count_{0};
  std::atomic<int64_t> failed_count_{0};

  // S3-specific metrics (merged from S3ClientMetrics)
  std::atomic<int64_t> multi_part_upload_created_{0};
  std::atomic<int64_t> multi_part_upload_finished_{0};
};

/// \brief Interface for filesystems that expose observable metrics
class Observable {
  public:
  virtual ~Observable() = default;

  /// \brief Get metrics for this filesystem
  /// \return Shared pointer to FilesystemMetrics, or nullptr if metrics are not available
  [[nodiscard]] virtual std::shared_ptr<FilesystemMetrics> GetMetrics() const = 0;
};

/// \brief Wrapper for InputStream to track read bytes
class MetricsInputStream : public arrow::io::InputStream {
  public:
  MetricsInputStream(std::shared_ptr<arrow::io::InputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, stream_->Read(nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, stream_->Read(nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Status Close() override { return stream_->Close(); }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }

  private:
  std::shared_ptr<arrow::io::InputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for RandomAccessFile to track read bytes
class MetricsRandomAccessFile : public arrow::io::RandomAccessFile {
  public:
  MetricsRandomAccessFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::shared_ptr<FilesystemMetrics> metrics)
      : file_(std::move(file)), metrics_(std::move(metrics)) {}

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, file_->Read(nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, file_->Read(nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, file_->ReadAt(position, nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, file_->ReadAt(position, nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Status Close() override { return file_->Close(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for OutputStream to track write bytes
class MetricsOutputStream : public arrow::io::OutputStream {
  public:
  MetricsOutputStream(std::shared_ptr<arrow::io::OutputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  arrow::Status Write(const void* data, int64_t nbytes) override {
    auto status = stream_->Write(data, nbytes);
    if (status.ok() && nbytes > 0) {
      metrics_->IncrementWriteBytes(nbytes);
    }
    return status;
  }

  arrow::Status Close() override { return stream_->Close(); }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  arrow::Status Flush() override { return stream_->Flush(); }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

}  // namespace milvus_storage
