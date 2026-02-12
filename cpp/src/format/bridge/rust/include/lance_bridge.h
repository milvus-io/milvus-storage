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
#include <stdexcept>
#include <arrow/c/abi.h>

#include "rust/cxx.h"
#include "rust-bridge/lib.h"

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

class LanceException : public std::runtime_error {
  public:
  explicit LanceException(const std::string& message) : std::runtime_error(message) {}
};

class BlockingFragmentReader;
class BlockingScanner;

/// Storage options for S3/cloud access
/// Keys correspond to Lance/object_store options:
///   - "aws_access_key_id" or "access_key_id"
///   - "aws_secret_access_key" or "secret_access_key"
///   - "aws_region" or "region"
///   - "aws_endpoint" or "endpoint"
///   - "allow_http" (set to "true" for non-SSL endpoints)
using LanceStorageOptions = std::unordered_map<std::string, std::string>;

/// Lance data storage format (file version)
enum class LanceDataStorageFormat : uint8_t {
  Legacy = 0,  // Lance 0.1 format, data in data/
  Stable = 1,  // Lance 2.0 format, data in _data/
};

class BlockingDataset {
  public:
  static std::shared_ptr<BlockingDataset> Open(const std::string& uri, const LanceStorageOptions& storage_options = {});

  static std::unique_ptr<BlockingDataset> OpenUnique(const std::string& uri,
                                                     const LanceStorageOptions& storage_options = {});

  static std::unique_ptr<BlockingDataset> WriteDataset(const std::string& uri,
                                                       struct ArrowArrayStream* stream,
                                                       const LanceStorageOptions& storage_options = {},
                                                       LanceDataStorageFormat format = LanceDataStorageFormat::Stable);

  explicit BlockingDataset(rust::Box<ffi::BlockingDataset> impl) : impl_(std::move(impl)) {}

  void WriteArrowArrayStream(struct ArrowArrayStream* stream);

  BlockingDataset(BlockingDataset&&) noexcept = default;
  BlockingDataset& operator=(BlockingDataset&&) noexcept = default;

  BlockingDataset(const BlockingDataset&) = delete;
  BlockingDataset& operator=(const BlockingDataset&) = delete;

  std::vector<uint64_t> GetAllFragmentIds() const;

  uint64_t GetFragmentRowCount(uint64_t fragment_id) const;

  // Dataset-level scan: create a scanner for projected columns
  std::unique_ptr<BlockingScanner> Scan(ArrowSchema& schema, uint32_t batch_size);

  // Dataset-level take: random access by global row indices
  ArrowArrayStream Take(const std::vector<int64_t>& indices, ArrowSchema& schema);

  const ffi::BlockingDataset& Impl() const { return *impl_; }

  private:
  rust::Box<ffi::BlockingDataset> impl_;
};

class BlockingFragmentReader {
  public:
  static std::unique_ptr<BlockingFragmentReader> Open(const BlockingDataset& dataset,
                                                      uint64_t fragment_id,
                                                      ArrowSchema& schema);

  explicit BlockingFragmentReader(rust::Box<ffi::BlockingFragmentReader> impl) : impl_(std::move(impl)) {}

  BlockingFragmentReader(BlockingFragmentReader&&) noexcept = default;
  BlockingFragmentReader& operator=(BlockingFragmentReader&&) noexcept = default;

  BlockingFragmentReader(const BlockingFragmentReader&) = delete;
  BlockingFragmentReader& operator=(const BlockingFragmentReader&) = delete;

  uint64_t RowCount() const;

  void TakeAsSingleBatch(const std::vector<int64_t>& indices, ArrowArray& out_array);

  ArrowArrayStream TakeAsStream(const std::vector<int64_t>& indices, uint32_t batch_size);

  ArrowArrayStream ReadAllAsStream(uint32_t batch_size);

  ArrowArrayStream ReadRangesAsStream(uint32_t row_range_start, uint32_t row_range_end, uint32_t batch_size);

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

  uint64_t CountRows() const;

  ArrowArrayStream OpenStream();

  private:
  rust::Box<ffi::BlockingScanner> impl_;
};

}  // namespace milvus_storage::lance
