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
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/future.h>

#include "milvus-storage/filesystem/metrics/buckets.h"
#include "milvus-storage/filesystem/metrics/error_classify.h"
#include "milvus-storage/filesystem/metrics/histogram.h"
#include "milvus-storage/filesystem/metrics/op_type.h"

namespace milvus_storage {

class ScopedOp;
class ScopedXfer;

/// \brief Labeled metrics registry for filesystem operations.
///
/// Every signal is keyed by OpType (and, for outcomes, OpStatus). Latency is
/// recorded per operation; payload size is recorded only for the three
/// transfer ops (Read, Write, MultipartUploadPart). A ScopedOp measures
/// exactly one backend request.
class FilesystemMetrics {
  public:
  struct OpStatsSnapshot {
    int64_t count_by_status[kStatusCount] = {};
    int64_t retry_count = 0;
    int64_t latency_sum_us = 0;
    int64_t latency_count = 0;
    int64_t latency_buckets[16] = {};
  };
  struct TransferSnapshot {
    int64_t bytes_total = 0;
    int64_t size_sum_bytes = 0;
    int64_t size_count = 0;
    int64_t size_buckets[14] = {};
  };
  struct Snapshot {
    OpStatsSnapshot ops[kOpTypeCount];
    TransferSnapshot transfers[kTransferCount];
    int64_t in_flight = 0;
    int64_t open_connections = 0;
    int64_t idle_connections = 0;
    int64_t pending_multipart_created = 0;
    int64_t pending_multipart_finished = 0;
  };

  FilesystemMetrics() = default;
  ~FilesystemMetrics() = default;

  /// \brief Begin a control/metadata operation. Records latency, status, and
  /// retries; increments in-flight for the scope's lifetime.
  [[nodiscard]] ScopedOp StartOp(OpType op);

  /// \brief Begin a data-transfer operation (Read, Write, MultipartUploadPart).
  /// Adds RecordBytes to the returned scope.
  [[nodiscard]] ScopedXfer StartTransfer(OpType op);

  /// \brief Best-effort connection-pool gauges, set by the backend client.
  void SetConnectionStats(int64_t open, int64_t idle) {
    open_connections_.store(open, std::memory_order_relaxed);
    idle_connections_.store(idle, std::memory_order_relaxed);
  }

  /// \brief Track pending multipart uploads (created - finished reveals leaks).
  void IncrementMultipartCreated() { pending_mp_created_.fetch_add(1, std::memory_order_relaxed); }
  void IncrementMultipartFinished() { pending_mp_finished_.fetch_add(1, std::memory_order_relaxed); }

  // ---- Live labeled queries over the registry ----
  /// \brief Count of one (op, status) cell.
  int64_t OpCount(OpType op, OpStatus status) const {
    return ops_[static_cast<int>(op)].count_by_status[static_cast<int>(status)].load(std::memory_order_relaxed);
  }
  /// \brief Count of an op across all statuses.
  int64_t OpCount(OpType op) const {
    int64_t total = 0;
    for (const auto& c : ops_[static_cast<int>(op)].count_by_status) {
      total += c.load(std::memory_order_relaxed);
    }
    return total;
  }
  /// \brief Total bytes moved by a transfer op (0 for non-transfer ops).
  int64_t TransferBytes(OpType op) const {
    int idx = TransferIndex(op);
    return idx < 0 ? 0 : transfers_[idx].bytes_total.load(std::memory_order_relaxed);
  }
  /// \brief Total failures (all non-Ok statuses across every op).
  int64_t FailedCount() const {
    int64_t failed = 0;
    for (const auto& slot : ops_) {
      for (int j = 0; j < kStatusCount; ++j) {
        if (j != static_cast<int>(OpStatus::Ok)) {
          failed += slot.count_by_status[j].load(std::memory_order_relaxed);
        }
      }
    }
    return failed;
  }
  int64_t InFlight() const { return in_flight_.load(std::memory_order_relaxed); }
  int64_t MultipartCreated() const { return pending_mp_created_.load(std::memory_order_relaxed); }
  int64_t MultipartFinished() const { return pending_mp_finished_.load(std::memory_order_relaxed); }

  Snapshot GetSnapshot() const {
    Snapshot s;
    for (int i = 0; i < kOpTypeCount; ++i) {
      const auto& src = ops_[i];
      auto& dst = s.ops[i];
      for (int j = 0; j < kStatusCount; ++j) {
        dst.count_by_status[j] = src.count_by_status[j].load(std::memory_order_relaxed);
      }
      dst.retry_count = src.retry_count.load(std::memory_order_relaxed);
      dst.latency_sum_us = src.latency.sum();
      dst.latency_count = src.latency.count();
      src.latency.CopyBuckets(dst.latency_buckets, 16);
    }
    for (int i = 0; i < kTransferCount; ++i) {
      s.transfers[i].bytes_total = transfers_[i].bytes_total.load(std::memory_order_relaxed);
      s.transfers[i].size_sum_bytes = transfers_[i].size.sum();
      s.transfers[i].size_count = transfers_[i].size.count();
      transfers_[i].size.CopyBuckets(s.transfers[i].size_buckets, 14);
    }
    s.in_flight = in_flight_.load(std::memory_order_relaxed);
    s.open_connections = open_connections_.load(std::memory_order_relaxed);
    s.idle_connections = idle_connections_.load(std::memory_order_relaxed);
    s.pending_multipart_created = pending_mp_created_.load(std::memory_order_relaxed);
    s.pending_multipart_finished = pending_mp_finished_.load(std::memory_order_relaxed);
    return s;
  }

  void Reset() {
    for (auto& slot : ops_) {
      for (auto& c : slot.count_by_status) {
        c.store(0, std::memory_order_relaxed);
      }
      slot.retry_count.store(0, std::memory_order_relaxed);
      slot.latency.Reset();
    }
    for (auto& t : transfers_) {
      t.bytes_total.store(0, std::memory_order_relaxed);
      t.size.Reset();
    }
    in_flight_.store(0, std::memory_order_relaxed);
    pending_mp_created_.store(0, std::memory_order_relaxed);
    pending_mp_finished_.store(0, std::memory_order_relaxed);
  }

  private:
  friend class ScopedOp;
  friend class ScopedXfer;

  struct OpSlot {
    std::atomic<int64_t> count_by_status[kStatusCount] = {};
    std::atomic<int64_t> retry_count{0};
    metrics::Histogram latency{metrics::kLatencyBoundsUs.data(), static_cast<int>(metrics::kLatencyBoundsUs.size())};
  };
  struct XferSlot {
    std::atomic<int64_t> bytes_total{0};
    metrics::Histogram size{metrics::kSizeBoundsBytes.data(), static_cast<int>(metrics::kSizeBoundsBytes.size())};
  };

  void Complete(OpType op, OpStatus status, int64_t latency_us, int64_t retries) {
    auto& slot = ops_[static_cast<int>(op)];
    slot.count_by_status[static_cast<int>(status)].fetch_add(1, std::memory_order_relaxed);
    slot.latency.Observe(latency_us);
    if (retries) {
      slot.retry_count.fetch_add(retries, std::memory_order_relaxed);
    }
    in_flight_.fetch_sub(1, std::memory_order_relaxed);
  }

  OpSlot ops_[kOpTypeCount];
  XferSlot transfers_[kTransferCount];
  std::atomic<int64_t> in_flight_{0};
  std::atomic<int64_t> open_connections_{0};
  std::atomic<int64_t> idle_connections_{0};
  std::atomic<int64_t> pending_mp_created_{0};
  std::atomic<int64_t> pending_mp_finished_{0};
};

/// \brief RAII scope measuring exactly one backend request.
class ScopedOp {
  public:
  ScopedOp(FilesystemMetrics* m, OpType op) : m_(m), op_(op), start_(std::chrono::steady_clock::now()) {
    m_->in_flight_.fetch_add(1, std::memory_order_relaxed);
  }
  ScopedOp(ScopedOp&& o) noexcept : m_(o.m_), op_(o.op_), start_(o.start_), status_(o.status_), retries_(o.retries_) {
    o.m_ = nullptr;
  }
  ScopedOp(const ScopedOp&) = delete;
  ScopedOp& operator=(const ScopedOp&) = delete;
  ScopedOp& operator=(ScopedOp&&) = delete;

  void RecordRetry(int n = 1) { retries_ += n; }
  void Fail(OpStatus s) { status_ = s; }

  // Abandon the scope without recording an operation (e.g. a zero-byte EOF
  // read that should not count). Still balances the in-flight gauge.
  void Cancel() {
    if (m_) {
      m_->in_flight_.fetch_sub(1, std::memory_order_relaxed);
      m_ = nullptr;
    }
  }

  ~ScopedOp() {
    if (!m_) {
      return;  // moved-from or cancelled
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_).count();
    m_->Complete(op_, status_, us, retries_);
  }

  protected:
  FilesystemMetrics* m_;
  OpType op_;
  std::chrono::steady_clock::time_point start_;
  OpStatus status_ = OpStatus::Ok;
  int64_t retries_ = 0;
};

/// \brief A ScopedOp for transfer ops, adding payload size recording.
class ScopedXfer : public ScopedOp {
  public:
  using ScopedOp::ScopedOp;

  void RecordBytes(int64_t n) {
    int idx = TransferIndex(op_);
    if (idx < 0) {
      return;
    }
    m_->transfers_[idx].bytes_total.fetch_add(n, std::memory_order_relaxed);
    m_->transfers_[idx].size.Observe(n);
  }
};

inline ScopedOp FilesystemMetrics::StartOp(OpType op) { return ScopedOp(this, op); }

inline ScopedXfer FilesystemMetrics::StartTransfer(OpType op) {
  assert(TransferIndex(op) >= 0 && "StartTransfer requires a transfer op");
  return ScopedXfer(this, op);
}

/// \brief Interface for filesystems that expose observable metrics
class Observable {
  public:
  virtual ~Observable() = default;

  /// \brief Get metrics for this filesystem
  /// \return Shared pointer to FilesystemMetrics, or nullptr if metrics are not available
  [[nodiscard]] virtual std::shared_ptr<FilesystemMetrics> GetMetrics() const = 0;
};

/// \brief Wrapper for InputStream to track read bytes and latency
class MetricsInputStream : public arrow::io::InputStream {
  public:
  MetricsInputStream(std::shared_ptr<arrow::io::InputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  // FileInterface
  arrow::Status Close() override { return stream_->Close(); }
  arrow::Status Abort() override { return stream_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  bool closed() const override { return stream_->closed(); }

  // Readable
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = stream_->Read(nbytes, out);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result > 0) {
      op.RecordBytes(*result);
    } else {
      op.Cancel();
    }
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = stream_->Read(nbytes);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result && (*result)->size() > 0) {
      op.RecordBytes((*result)->size());
    } else {
      op.Cancel();
    }
    return result;
  }

  const arrow::io::IOContext& io_context() const override { return stream_->io_context(); }

  // InputStream
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return stream_->Peek(nbytes); }
  bool supports_zero_copy() const override { return stream_->supports_zero_copy(); }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return stream_->ReadMetadata();
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return stream_->ReadMetadataAsync(io_context);
  }

  private:
  std::shared_ptr<arrow::io::InputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for RandomAccessFile to track read bytes and latency
class MetricsRandomAccessFile : public arrow::io::RandomAccessFile {
  public:
  MetricsRandomAccessFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::shared_ptr<FilesystemMetrics> metrics)
      : file_(std::move(file)), metrics_(std::move(metrics)) {}

  // FileInterface
  arrow::Status Close() override { return file_->Close(); }
  arrow::Status Abort() override { return file_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  bool closed() const override { return file_->closed(); }

  // Readable
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = file_->Read(nbytes, out);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result > 0) {
      op.RecordBytes(*result);
    } else {
      op.Cancel();
    }
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = file_->Read(nbytes);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result && (*result)->size() > 0) {
      op.RecordBytes((*result)->size());
    } else {
      op.Cancel();
    }
    return result;
  }

  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }

  // InputStream
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return file_->ReadMetadata();
  }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return file_->ReadMetadataAsync(io_context);
  }

  // Seekable
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }

  // RandomAccessFile
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = file_->ReadAt(position, nbytes, out);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result > 0) {
      op.RecordBytes(*result);
    } else {
      op.Cancel();
    }
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    auto op = metrics_->StartTransfer(OpType::Read);
    auto result = file_->ReadAt(position, nbytes);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }
    if (*result && (*result)->size() > 0) {
      op.RecordBytes((*result)->size());
    } else {
      op.Cancel();
    }
    return result;
  }

  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    auto op = std::make_shared<ScopedXfer>(metrics_->StartTransfer(OpType::Read));
    return file_->ReadAsync(io_context, position, nbytes)
        .Then(
            [op](const std::shared_ptr<arrow::Buffer>& buffer) {
              if (buffer && buffer->size() > 0) {
                op->RecordBytes(buffer->size());
              } else {
                op->Cancel();
              }
              return buffer;
            },
            [op](const arrow::Status& status) {
              op->Fail(ClassifyArrowStatus(status));
              return arrow::Result<std::shared_ptr<arrow::Buffer>>(status);
            });
  }

  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
      const arrow::io::IOContext& io_context, const std::vector<arrow::io::ReadRange>& ranges) override {
    auto futures = file_->ReadManyAsync(io_context, ranges);
    for (auto& future : futures) {
      auto op = std::make_shared<ScopedXfer>(metrics_->StartTransfer(OpType::Read));
      future = future.Then(
          [op](const std::shared_ptr<arrow::Buffer>& buffer) {
            if (buffer && buffer->size() > 0) {
              op->RecordBytes(buffer->size());
            } else {
              op->Cancel();
            }
            return buffer;
          },
          [op](const arrow::Status& status) {
            op->Fail(ClassifyArrowStatus(status));
            return arrow::Result<std::shared_ptr<arrow::Buffer>>(status);
          });
    }
    return futures;
  }

  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for OutputStream to track write bytes and latency
class MetricsOutputStream : public arrow::io::OutputStream {
  public:
  MetricsOutputStream(std::shared_ptr<arrow::io::OutputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  // FileInterface
  arrow::Status Close() override { return stream_->Close(); }
  arrow::Status Abort() override { return stream_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  bool closed() const override { return stream_->closed(); }

  // Writable
  arrow::Status Write(const void* data, int64_t nbytes) override {
    auto op = metrics_->StartTransfer(OpType::Write);
    auto status = stream_->Write(data, nbytes);
    if (!status.ok()) {
      op.Fail(ClassifyArrowStatus(status));
      return status;
    }
    if (nbytes > 0) {
      op.RecordBytes(nbytes);
    } else {
      op.Cancel();
    }
    return status;
  }

  arrow::Status Write(const std::shared_ptr<arrow::Buffer>& data) override {
    auto op = metrics_->StartTransfer(OpType::Write);
    auto status = stream_->Write(data);
    if (!status.ok()) {
      op.Fail(ClassifyArrowStatus(status));
      return status;
    }
    if (data && data->size() > 0) {
      op.RecordBytes(data->size());
    } else {
      op.Cancel();
    }
    return status;
  }

  arrow::Status Flush() override { return stream_->Flush(); }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

}  // namespace milvus_storage
