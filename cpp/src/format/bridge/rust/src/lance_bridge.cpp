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

#include <memory>
#include <string_view>
#include <utility>

#include <arrow/record_batch.h>

#include "bridge_util.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

namespace milvus_storage::lance {
namespace {

class LanceErrorTranslatingReader final : public arrow::RecordBatchReader {
  public:
  explicit LanceErrorTranslatingReader(std::shared_ptr<arrow::RecordBatchReader> inner) : inner_(std::move(inner)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return inner_->schema(); }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  arrow::Status Close() override;

  private:
  std::shared_ptr<arrow::RecordBatchReader> inner_;
};

arrow::Status LanceErrorTranslatingReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  auto status = inner_->ReadNext(batch);
  return MakeBridgeErrorStatus("Failed to read Lance record batch", status);
}

arrow::Status LanceErrorTranslatingReader::Close() {
  return MakeBridgeErrorStatus("Failed to close Lance record batch reader", inner_->Close());
}

}  // namespace

namespace internal {

std::shared_ptr<arrow::RecordBatchReader> WrapLanceRecordBatchReader(std::shared_ptr<arrow::RecordBatchReader> inner) {
  return std::make_shared<LanceErrorTranslatingReader>(std::move(inner));
}

}  // namespace internal

void ReplaceLanceRuntime(uint32_t num_threads) {}

using milvus_storage::ConvertStorageOptions;

arrow::Result<std::shared_ptr<BlockingDataset>> BlockingDataset::Open(
    const std::string& uri,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const StorageOptions& read_options,
    uint64_t version) {
  if (!filesystem) {
    return arrow::Status::Invalid("BlockingDataset::Open requires a non-null filesystem");
  }
  return CatchRustResult<std::shared_ptr<BlockingDataset>>("Failed to open Lance dataset", [&]() {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(read_options, keys, values);
    auto filesystem_lease = std::make_shared<FileSystemWrapper>(filesystem);
    auto impl = ffi::open_dataset(std::move(filesystem_lease), rust::Str(uri.data(), uri.length()), std::move(keys),
                                  std::move(values), version);
    return std::make_shared<BlockingDataset>(std::move(impl));
  });
}

arrow::Result<uint64_t> BlockingDataset::ResolveLatestVersion(const std::string& uri,
                                                              const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
                                                              const StorageOptions& read_options) {
  if (!filesystem) {
    return arrow::Status::Invalid("BlockingDataset::ResolveLatestVersion requires a non-null filesystem");
  }
  return CatchRustResult<uint64_t>("Failed to resolve latest Lance dataset version", [&]() {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(read_options, keys, values);
    auto filesystem_lease = std::make_shared<FileSystemWrapper>(filesystem);
    return ffi::resolve_latest_dataset_version(std::move(filesystem_lease), rust::Str(uri.data(), uri.length()),
                                               std::move(keys), std::move(values));
  });
}

uint64_t BlockingDataset::Version() const { return impl_->version(); }

arrow::Result<std::vector<uint64_t>> BlockingDataset::WriteDataset(const std::string& uri,
                                                                   struct ArrowArrayStream* stream,
                                                                   const StorageOptions& storage_options,
                                                                   LanceDataStorageFormat format) {
  return CatchRustResult<std::vector<uint64_t>>("Failed to write Lance dataset", [&]() {
    ArrowCDataReleaseGuard stream_guard(stream);
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    auto ffi_format = static_cast<ffi::LanceDataStorageFormat>(format);
    auto fragment_ids = ffi::write_dataset(rust::Str(uri.data(), uri.length()), reinterpret_cast<uint8_t*>(stream),
                                           std::move(keys), std::move(values), ffi_format);
    stream_guard.Disarm();
    return std::vector<uint64_t>(fragment_ids.begin(), fragment_ids.end());
  });
}

arrow::Status BlockingDataset::DeleteRows(const std::string& uri,
                                          const std::string& predicate,
                                          const StorageOptions& native_options) {
  return CatchRustStatus("Failed to delete Lance rows", [&]() {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(native_options, keys, values);
    ffi::delete_rows(rust::Str(uri.data(), uri.length()), rust::Str(predicate.data(), predicate.length()),
                     std::move(keys), std::move(values));
  });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::GetAllFragmentIds() const {
  return CatchRustResult<std::vector<uint64_t>>("Failed to get Lance fragment IDs", [&]() {
    auto fragment_ids = impl_->get_all_fragment_ids();
    return std::vector<uint64_t>(fragment_ids.begin(), fragment_ids.end());
  });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::GetFragmentDeletionPositions(uint64_t fragment_id) const {
  return CatchRustResult<std::vector<uint64_t>>("Failed to get Lance fragment deletion positions", [&]() {
    auto positions = ffi::get_fragment_deletion_positions(*impl_, fragment_id);
    return std::vector<uint64_t>(positions.begin(), positions.end());
  });
}

arrow::Result<uint64_t> BlockingDataset::GetFragmentPhysicalRowCount(uint64_t fragment_id) const {
  return CatchRustResult<uint64_t>("Failed to get Lance fragment physical row count",
                                   [&]() { return ffi::get_fragment_physical_row_count(*impl_, fragment_id); });
}

arrow::Result<uint64_t> BlockingDataset::GetFragmentRowCount(uint64_t fragment_id) const {
  return CatchRustResult<uint64_t>("Failed to get Lance fragment row count",
                                   [&]() { return ffi::get_fragment_row_count(*impl_, fragment_id); });
}

arrow::Result<std::vector<uint64_t>> BlockingDataset::EstimateFragmentColumnMemory(uint64_t fragment_id) const {
  auto result = CatchRustResult<std::vector<uint64_t>>([&]() {
    auto estimates = ffi::estimate_fragment_column_memory(*impl_, fragment_id);
    std::vector<uint64_t> memory_sizes;
    memory_sizes.reserve(estimates.size());
    for (const auto& estimate : estimates) {
      memory_sizes.push_back(estimate.memory_size);
    }
    return memory_sizes;
  });
  if (!result.ok()) {
    return arrow::Status::NotImplemented("Lance column memory size estimation is not available: ",
                                         result.status().message());
  }
  return result;
}

arrow::Result<uint64_t> BlockingDataset::EstimateFragmentMemory(uint64_t fragment_id) const {
  return CatchRustResult<uint64_t>("Failed to estimate Lance fragment memory",
                                   [&]() { return ffi::estimate_fragment_memory(*impl_, fragment_id); });
}

arrow::Status BlockingDataset::GetFragmentSchema(uint64_t fragment_id, ArrowSchema& out_schema) const {
  out_schema = {};
  return CatchRustStatus("Failed to get Lance fragment schema", [&]() {
    ArrowCDataReleaseGuard schema_guard(&out_schema);
    ffi::get_fragment_schema(*impl_, fragment_id, reinterpret_cast<uint8_t*>(&out_schema));
    schema_guard.Disarm();
  });
}

arrow::Result<std::unique_ptr<BlockingFragmentReader>> BlockingFragmentReader::Open(const BlockingDataset& dataset,
                                                                                    uint64_t fragment_id,
                                                                                    ArrowSchema& schema) {
  return CatchRustResult<std::unique_ptr<BlockingFragmentReader>>("Failed to open Lance fragment reader", [&]() {
    ArrowCDataReleaseGuard schema_guard(&schema);
    auto impl = ffi::open_fragment_reader(dataset.Impl(), fragment_id, reinterpret_cast<uint8_t*>(&schema));
    schema_guard.Disarm();
    return std::make_unique<BlockingFragmentReader>(std::move(impl));
  });
}

arrow::Result<uint64_t> BlockingFragmentReader::RowCount() const {
  return CatchRustResult<uint64_t>("Failed to get Lance fragment row count", [&]() { return impl_->number_of_rows(); });
}

arrow::Status BlockingFragmentReader::TakeAsSingleBatch(const std::vector<int64_t>& indices, ArrowArray& out_array) {
  out_array = {};
  return CatchRustStatus("Failed to take Lance rows as a single batch", [&]() {
    ArrowCDataReleaseGuard array_guard(&out_array);
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_single_batch(indices_slice, reinterpret_cast<uint8_t*>(&out_array));
    array_guard.Disarm();
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::TakeAsStream(const std::vector<int64_t>& indices,
                                                                     uint32_t batch_size) {
  return CatchRustResult<ArrowArrayStream>("Failed to take Lance rows as a stream", [&]() {
    ArrowArrayStream stream{};
    ArrowCDataReleaseGuard stream_guard(&stream);
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_stream(indices_slice, batch_size, reinterpret_cast<uint8_t*>(&stream));
    stream_guard.Disarm();
    return stream;
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::ReadAllAsStream(uint32_t batch_size) {
  return CatchRustResult<ArrowArrayStream>("Failed to read Lance fragment", [&]() {
    ArrowArrayStream stream{};
    ArrowCDataReleaseGuard stream_guard(&stream);
    impl_->read_all_as_stream(batch_size, reinterpret_cast<uint8_t*>(&stream));
    stream_guard.Disarm();
    return stream;
  });
}

arrow::Result<ArrowArrayStream> BlockingFragmentReader::ReadRangesAsStream(uint32_t row_range_start,
                                                                           uint32_t row_range_end,
                                                                           uint32_t batch_size) {
  return CatchRustResult<ArrowArrayStream>("Failed to read Lance row range", [&]() {
    ArrowArrayStream stream{};
    ArrowCDataReleaseGuard stream_guard(&stream);
    impl_->read_ranges_as_stream(row_range_start, row_range_end, batch_size, reinterpret_cast<uint8_t*>(&stream));
    stream_guard.Disarm();
    return stream;
  });
}

arrow::Result<std::unique_ptr<BlockingScanner>> BlockingDataset::Scan(ArrowSchema& schema, uint32_t batch_size) {
  return CatchRustResult<std::unique_ptr<BlockingScanner>>("Failed to create Lance scanner", [&]() {
    ArrowCDataReleaseGuard schema_guard(&schema);
    auto impl = ffi::create_scanner(*impl_, reinterpret_cast<uint8_t*>(&schema), batch_size);
    schema_guard.Disarm();
    return std::make_unique<BlockingScanner>(std::move(impl));
  });
}

#ifdef BUILD_GTEST
arrow::Result<LanceIOStats> BlockingDataset::IOStatsIncremental() {
  return CatchRustResult<LanceIOStats>("Failed to get Lance IO statistics", [&]() {
    auto stats = impl_->io_stats_incremental();
    return LanceIOStats{stats.read_iops, stats.read_bytes};
  });
}
#endif  // BUILD_GTEST

arrow::Result<ArrowArrayStream> BlockingDataset::Take(const std::vector<int64_t>& indices, ArrowSchema& schema) {
  return CatchRustResult<ArrowArrayStream>("Failed to take Lance dataset rows", [&]() {
    ArrowCDataReleaseGuard schema_guard(&schema);
    ArrowArrayStream stream{};
    ArrowCDataReleaseGuard stream_guard(&stream);
    std::vector<uint64_t> uint64_indices(indices.begin(), indices.end());
    rust::Slice<const uint64_t> indices_slice(uint64_indices.data(), uint64_indices.size());
    ffi::dataset_take(*impl_, indices_slice, reinterpret_cast<uint8_t*>(&schema), reinterpret_cast<uint8_t*>(&stream));
    schema_guard.Disarm();
    stream_guard.Disarm();
    return stream;
  });
}

arrow::Result<uint64_t> BlockingScanner::CountRows() const {
  return CatchRustResult<uint64_t>("Failed to count Lance scanner rows", [&]() { return impl_->count_rows(); });
}

arrow::Result<ArrowArrayStream> BlockingScanner::OpenStream() {
  return CatchRustResult<ArrowArrayStream>("Failed to open Lance scanner stream", [&]() {
    ArrowArrayStream stream{};
    ArrowCDataReleaseGuard stream_guard(&stream);
    impl_->open_stream(reinterpret_cast<uint8_t*>(&stream));
    stream_guard.Disarm();
    return stream;
  });
}

}  // namespace milvus_storage::lance
