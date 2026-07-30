// Copyright 2023 Zilliz
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

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <arrow/c/abi.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "rust/cxx.h"
#include "rust-bridge/lib.h"

// Error model: every fallible API returns arrow::Result / arrow::Status.
// Errors coming out of the Rust bridge carry a classification marker (see
// bridge_error.h) that is decoded into a structured status here — not-found
// surfaces with an ENOENT detail, transients with an ExtendStatusDetail,
// corruption as Status::Invalid — instead of the previous string-only
// LanceException, which leaked exceptions out of the library and collapsed
// every failure class into one bucket.

namespace milvus_storage::lance {

/// Replace the global Lance tokio runtime with a new one using the specified number of worker threads.
///
/// **WARNING: DANGEROUS. For benchmarks/tests ONLY. DO NOT use in production.**
///
/// Caller MUST guarantee ALL of the following before calling:
/// - No Lance operations are in-flight (no pending scans, reads, writes, or any other async work).
/// - No references to the old runtime are held anywhere.
/// - No other thread is concurrently calling any Lance API.
///
/// Violating any of the above leads to undefined behavior (use-after-free, data races).
void ReplaceLanceRuntime(uint32_t num_threads);

class BlockingFragmentReader;
class BlockingScanner;

/// Lance IO statistics (read-and-reset semantics)
struct LanceIOStats {
  uint64_t read_iops = 0;
  uint64_t read_bytes = 0;
};

/// Storage options type for S3/cloud access (key-value pairs).
using StorageOptions = std::unordered_map<std::string, std::string>;

/// Lance data storage format (file version)
enum class LanceDataStorageFormat : uint8_t {
  Legacy = 0,  // Lance 0.1 format, data in data/
  V2_1 = 1,
  Stable = V2_1,  // Backward-compatible name for the default format.
  V2_2 = 2,
  V2_3 = 3,
};

class BlockingDataset {
  public:
  static arrow::Result<std::shared_ptr<BlockingDataset>> Open(const std::string& uri,
                                                              const StorageOptions& storage_options = {});

  static arrow::Result<std::unique_ptr<BlockingDataset>> OpenUnique(const std::string& uri,
                                                                    const StorageOptions& storage_options = {});

  static arrow::Result<std::unique_ptr<BlockingDataset>> WriteDataset(
      const std::string& uri,
      struct ArrowArrayStream* stream,
      const StorageOptions& storage_options = {},
      LanceDataStorageFormat format = LanceDataStorageFormat::Stable);

  explicit BlockingDataset(rust::Box<ffi::BlockingDataset> impl) : impl_(std::move(impl)) {}

  arrow::Status WriteArrowArrayStream(struct ArrowArrayStream* stream);

  BlockingDataset(BlockingDataset&&) noexcept = default;
  BlockingDataset& operator=(BlockingDataset&&) noexcept = default;

  BlockingDataset(const BlockingDataset&) = delete;
  BlockingDataset& operator=(const BlockingDataset&) = delete;

  arrow::Status DeleteRows(const std::string& predicate);

  arrow::Result<std::vector<uint64_t>> GetAllFragmentIds() const;

  arrow::Result<std::vector<uint64_t>> GetFragmentDeletionPositions(uint64_t fragment_id) const;

  arrow::Result<uint64_t> GetFragmentPhysicalRowCount(uint64_t fragment_id) const;

  arrow::Result<uint64_t> GetFragmentRowCount(uint64_t fragment_id) const;

  // Top-level dataset columns in schema order; returns NotImplemented when estimation is unavailable.
  arrow::Result<std::vector<uint64_t>> EstimateFragmentColumnMemory(uint64_t fragment_id) const;

  // Best-effort: returns 0 when estimation is unavailable.
  uint64_t EstimateFragmentMemory(uint64_t fragment_id) const;

  // Lance 7 exposes the current dataset schema through FileFragment::schema().
  // It can include evolved nullable columns that are not physically stored in this fragment.
  arrow::Status GetFragmentSchema(uint64_t fragment_id, ArrowSchema& out_schema) const;

  // Dataset-level scan: create a scanner for projected columns
  arrow::Result<std::unique_ptr<BlockingScanner>> Scan(ArrowSchema& schema, uint32_t batch_size);

  // Dataset-level take: random access by global row indices
  arrow::Result<ArrowArrayStream> Take(const std::vector<int64_t>& indices, ArrowSchema& schema);

  /// Read and reset IO statistics for this dataset's object store.
  /// Best-effort: returns zeroes when statistics are unavailable.
  LanceIOStats IOStatsIncremental();

  const ffi::BlockingDataset& Impl() const { return *impl_; }

  private:
  rust::Box<ffi::BlockingDataset> impl_;
};

class BlockingFragmentReader {
  public:
  static arrow::Result<std::unique_ptr<BlockingFragmentReader>> Open(const BlockingDataset& dataset,
                                                                     uint64_t fragment_id,
                                                                     ArrowSchema& schema);

  explicit BlockingFragmentReader(rust::Box<ffi::BlockingFragmentReader> impl) : impl_(std::move(impl)) {}

  BlockingFragmentReader(BlockingFragmentReader&&) noexcept = default;
  BlockingFragmentReader& operator=(BlockingFragmentReader&&) noexcept = default;

  BlockingFragmentReader(const BlockingFragmentReader&) = delete;
  BlockingFragmentReader& operator=(const BlockingFragmentReader&) = delete;

  arrow::Result<uint64_t> RowCount() const;

  arrow::Status TakeAsSingleBatch(const std::vector<int64_t>& indices, ArrowArray& out_array);

  arrow::Result<ArrowArrayStream> TakeAsStream(const std::vector<int64_t>& indices, uint32_t batch_size);

  arrow::Result<ArrowArrayStream> ReadAllAsStream(uint32_t batch_size);

  arrow::Result<ArrowArrayStream> ReadRangesAsStream(uint32_t row_range_start,
                                                     uint32_t row_range_end,
                                                     uint32_t batch_size);

  private:
  rust::Box<ffi::BlockingFragmentReader> impl_;
};

class BlockingScanner {
  public:
  explicit BlockingScanner(rust::Box<ffi::BlockingScanner> impl) : impl_(std::move(impl)) {}

  BlockingScanner(BlockingScanner&&) noexcept = default;
  BlockingScanner& operator=(BlockingScanner&&) noexcept = default;

  BlockingScanner(const BlockingScanner&) = delete;
  BlockingScanner& operator=(const BlockingScanner&) = delete;

  arrow::Result<uint64_t> CountRows() const;

  arrow::Result<ArrowArrayStream> OpenStream();

  private:
  rust::Box<ffi::BlockingScanner> impl_;
};

}  // namespace milvus_storage::lance
