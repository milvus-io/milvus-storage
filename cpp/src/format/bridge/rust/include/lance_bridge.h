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
#include <stdexcept>
#include <arrow/c/abi.h>

#include <arrow/c/abi.h>

#include "rust/cxx.h"
#include "rust-bridge/lib.h"

namespace milvus_storage::lance {

class LanceException : public std::runtime_error {
  public:
  explicit LanceException(const std::string& message) : std::runtime_error(message) {}
};

class BlockingFragmentReader;

class BlockingDataset {
  public:
  static std::shared_ptr<BlockingDataset> Open(const std::string& uri);

  static std::unique_ptr<BlockingDataset> OpenUnique(const std::string& uri);

  static std::unique_ptr<BlockingDataset> WriteDataset(const std::string& uri, struct ArrowArrayStream* stream);

  explicit BlockingDataset(rust::Box<ffi::BlockingDataset> impl) : impl_(std::move(impl)) {}

  void WriteArrowArrayStream(struct ArrowArrayStream* stream);

  BlockingDataset(BlockingDataset&&) noexcept = default;
  BlockingDataset& operator=(BlockingDataset&&) noexcept = default;

  BlockingDataset(const BlockingDataset&) = delete;
  BlockingDataset& operator=(const BlockingDataset&) = delete;

  std::vector<uint64_t> GetAllFragmentIds() const;

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

}  // namespace milvus_storage::lance
