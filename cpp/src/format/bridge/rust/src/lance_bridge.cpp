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

#include "lance_bridge.h"
#include "bridge_error.h"
#include "bridge_util.h"

#include <memory>
#include <utility>

namespace milvus_storage::lance {

void ReplaceLanceRuntime(uint32_t num_threads) {}

using milvus_storage::ConvertStorageOptions;
// Both guards live in bridge_error.h now: they clear the side channel, run the
// call, and on a cxx exception prefer the classification the Rust side recorded
// over anything parsed out of the message.
using milvus_storage::bridge::CatchBridgeError;
using milvus_storage::bridge::CatchBridgeStatus;

arrow::Result<std::shared_ptr<BlockingDataset>> BlockingDataset::Open(const std::string& uri,
                                                                      const StorageOptions& storage_options) {
  return CatchBridgeError([&] {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    return std::make_shared<BlockingDataset>(
        ffi::open_dataset(rust::Str(uri.data(), uri.length()), std::move(keys), std::move(values)));
  });
}

arrow::Result<std::unique_ptr<BlockingDataset>> BlockingDataset::OpenUnique(const std::string& uri,
                                                                            const StorageOptions& storage_options) {
  return CatchBridgeError([&] {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    return std::make_unique<BlockingDataset>(
        ffi::open_dataset(rust::Str(uri.data(), uri.length()), std::move(keys), std::move(values)));
  });
}

arrow::Result<std::unique_ptr<BlockingDataset>> BlockingDataset::WriteDataset(const std::string& uri,
                                                                              struct ArrowArrayStream* stream,
                                                                              const StorageOptions& storage_options,
                                                                              LanceDataStorageFormat format) {
  return CatchBridgeError([&] {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    auto ffi_format = static_cast<ffi::LanceDataStorageFormat>(format);
    return std::make_unique<BlockingDataset>(ffi::write_dataset(rust::Str(uri.data(), uri.length()),
                                                                reinterpret_cast<uint8_t*>(stream), std::move(keys),
                                                                std::move(values), ffi_format));
  });
}

arrow::Status BlockingDataset::DeleteRows(const std::string& predicate) {
  return CatchBridgeStatus([&] { ffi::dataset_delete_rows(*impl_, rust::Str(predicate.data(), predicate.length())); });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::GetAllFragmentIds() const {
  return CatchBridgeError([&] {
    auto fragment_ids = impl_->get_all_fragment_ids();
    return std::vector<uint64_t>{fragment_ids.begin(), fragment_ids.end()};
  });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::GetFragmentDeletionPositions(uint64_t fragment_id) const {
  return CatchBridgeError([&] {
    auto positions = ffi::get_fragment_deletion_positions(*impl_, fragment_id);
    return std::vector<uint64_t>{positions.begin(), positions.end()};
  });
}

arrow::Result<uint64_t> BlockingDataset::GetFragmentPhysicalRowCount(uint64_t fragment_id) const {
  return CatchBridgeError([&] { return ffi::get_fragment_physical_row_count(*impl_, fragment_id); });
}

arrow::Result<uint64_t> BlockingDataset::GetFragmentRowCount(uint64_t fragment_id) const {
  return CatchBridgeError([&] { return ffi::get_fragment_row_count(*impl_, fragment_id); });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::EstimateFragmentColumnMemory(uint64_t fragment_id) const {
  return CatchBridgeError([&] {
    auto estimates = ffi::estimate_fragment_column_memory(*impl_, fragment_id);
    std::vector<uint64_t> memory_sizes;
    memory_sizes.reserve(estimates.size());
    for (const auto& estimate : estimates) {
      memory_sizes.push_back(estimate.memory_size);
    }
    return memory_sizes;
  });
}

arrow::Result<uint64_t> BlockingDataset::EstimateFragmentMemory(uint64_t fragment_id) const {
  return CatchBridgeError([&] { return ffi::estimate_fragment_memory(*impl_, fragment_id); });
}

arrow::Status BlockingDataset::GetFragmentSchema(uint64_t fragment_id, ArrowSchema& out_schema) const {
  return CatchBridgeStatus(
      [&] { ffi::get_fragment_schema(*impl_, fragment_id, reinterpret_cast<uint8_t*>(&out_schema)); });
}

arrow::Status BlockingDataset::WriteArrowArrayStream(struct ArrowArrayStream* stream) {
  return CatchBridgeStatus([&] { impl_->write_stream(reinterpret_cast<uint8_t*>(stream)); });
}

arrow::Result<std::unique_ptr<BlockingFragmentReader>> BlockingFragmentReader::Open(const BlockingDataset& dataset,
                                                                                    uint64_t fragment_id,
                                                                                    ArrowSchema& schema) {
  return CatchBridgeError([&] {
    auto impl = ffi::open_fragment_reader(dataset.Impl(), fragment_id, reinterpret_cast<uint8_t*>(&schema));
    return std::make_unique<BlockingFragmentReader>(std::move(impl));
  });
}

arrow::Result<uint64_t> BlockingFragmentReader::RowCount() const {
  return CatchBridgeError([&] { return impl_->number_of_rows(); });
}

arrow::Status BlockingFragmentReader::TakeAsSingleBatch(const std::vector<int64_t>& indices, ArrowArray& out_array) {
  return CatchBridgeStatus([&] {
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_single_batch(indices_slice, reinterpret_cast<uint8_t*>(&out_array));
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::TakeAsStream(const std::vector<int64_t>& indices,
                                                                     uint32_t batch_size) {
  return CatchBridgeError([&] {
    ArrowArrayStream stream;
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_stream(indices_slice, batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::ReadAllAsStream(uint32_t batch_size) {
  return CatchBridgeError([&] {
    ArrowArrayStream stream;
    impl_->read_all_as_stream(batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::ReadRangesAsStream(uint32_t row_range_start,
                                                                           uint32_t row_range_end,
                                                                           uint32_t batch_size) {
  return CatchBridgeError([&] {
    ArrowArrayStream stream;
    impl_->read_ranges_as_stream(row_range_start, row_range_end, batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

arrow::Result<std::unique_ptr<BlockingScanner>> BlockingDataset::Scan(ArrowSchema& schema, uint32_t batch_size) {
  return CatchBridgeError([&] {
    auto impl = ffi::create_scanner(*impl_, reinterpret_cast<uint8_t*>(&schema), batch_size);
    return std::make_unique<BlockingScanner>(std::move(impl));
  });
}

#ifdef BUILD_GTEST
LanceIOStats BlockingDataset::IOStatsIncremental() {
  try {
    auto stats = impl_->io_stats_incremental();
    return {stats.read_iops, stats.read_bytes};
  } catch (const rust::cxxbridge1::Error&) {
    return {};
  }
}
#endif  // BUILD_GTEST

arrow::Result<ArrowArrayStream> BlockingDataset::Take(const std::vector<int64_t>& indices, ArrowSchema& schema) {
  return CatchBridgeError([&] {
    ArrowArrayStream stream;
    std::vector<uint64_t> uint64_indices(indices.begin(), indices.end());
    rust::Slice<const uint64_t> indices_slice(uint64_indices.data(), uint64_indices.size());
    ffi::dataset_take(*impl_, indices_slice, reinterpret_cast<uint8_t*>(&schema), reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

arrow::Result<uint64_t> BlockingScanner::CountRows() const {
  return CatchBridgeError([&] { return impl_->count_rows(); });
}

arrow::Result<ArrowArrayStream> BlockingScanner::OpenStream() {
  return CatchBridgeError([&] {
    ArrowArrayStream stream;
    impl_->open_stream(reinterpret_cast<uint8_t*>(&stream));
    return stream;
  });
}

}  // namespace milvus_storage::lance
