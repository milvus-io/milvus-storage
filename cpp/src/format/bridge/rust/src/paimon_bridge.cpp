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

#include "paimon_bridge.h"

#include <atomic>

#include "bridge_util.h"

namespace milvus_storage::paimon {
namespace {

void ConvertStrings(const std::vector<std::string>& source, rust::Vec<rust::String>& target) {
  target.reserve(source.size());
  for (const auto& value : source) {
    target.emplace_back(value);
  }
}

std::atomic<uint64_t> data_split_reader_open_count{0};
std::atomic<uint64_t> data_split_stream_open_count{0};

}  // namespace

uint64_t GetPaimonDataSplitReaderOpenCount() { return data_split_reader_open_count.load(std::memory_order_relaxed); }

uint64_t GetPaimonDataSplitStreamOpenCount() { return data_split_stream_open_count.load(std::memory_order_relaxed); }

std::vector<PaimonFileInfo> PlanFiles(const std::string& table_location,
                                      int64_t snapshot_id,
                                      const std::string& scan_mode,
                                      const StorageOptions& storage_options) {
  try {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto planned = ffi::paimon_plan_files(rust::Str(table_location), snapshot_id, rust::Str(scan_mode), std::move(keys),
                                          std::move(values));
    std::vector<PaimonFileInfo> result;
    result.reserve(planned.size());
    for (const auto& info : planned) {
      result.push_back(PaimonFileInfo{std::string(info.path), std::string(info.read_path),
                                      std::string(info.route_reason), info.file_size,
                                      std::string(info.metadata_json)});
    }
    return result;
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

std::vector<uint64_t> ReadDeletionVector(const std::string& path,
                                         uint64_t offset,
                                         uint64_t length,
                                         int64_t expected_cardinality,
                                         const StorageOptions& storage_options) {
  try {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto positions = ffi::paimon_read_deletion_vector(rust::Str(path), offset, length, expected_cardinality,
                                                      std::move(keys), std::move(values));
    return {positions.begin(), positions.end()};
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

PaimonTestTableInfo CreateTestTable(const std::string& table_location,
                                    uint64_t num_rows,
                                    const std::string& mode,
                                    const std::vector<int64_t>& deleted_positions,
                                    const StorageOptions& storage_options,
                                    const std::string& file_format,
                                    uint32_t dimension) {
  try {
    rust::Vec<int64_t> positions;
    positions.reserve(deleted_positions.size());
    for (auto position : deleted_positions) {
      positions.push_back(position);
    }
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto info =
        ffi::paimon_create_test_table(rust::Str(table_location), num_rows, rust::Str(mode), std::move(positions),
                                      std::move(keys), std::move(values), rust::Str(file_format), dimension);
    return PaimonTestTableInfo{
        std::string(info.table_location), info.snapshot_id, {info.snapshot_ids.begin(), info.snapshot_ids.end()}};
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

std::unique_ptr<BlockingPaimonDataSplitReader> BlockingPaimonDataSplitReader::Open(
    const std::string& metadata_json,
    const std::string& expected_table_location,
    const StorageOptions& storage_options) {
  try {
    rust::Vec<rust::String> keys;
    rust::Vec<rust::String> values;
    ConvertStorageOptions(storage_options, keys, values);
    auto reader = std::make_unique<BlockingPaimonDataSplitReader>(ffi::paimon_open_data_split_reader(
        rust::Str(metadata_json), rust::Str(expected_table_location), std::move(keys), std::move(values)));
    data_split_reader_open_count.fetch_add(1, std::memory_order_relaxed);
    return reader;
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

void BlockingPaimonDataSplitReader::ExportSchema(ArrowSchema* schema) const {
  if (schema == nullptr) {
    throw PaimonException("cannot export Paimon schema into a null pointer");
  }
  try {
    impl_->export_schema(reinterpret_cast<uint8_t*>(schema));
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

ArrowArrayStream BlockingPaimonDataSplitReader::OpenStream(const std::vector<std::string>& projected_columns) const {
  try {
    rust::Vec<rust::String> columns;
    ConvertStrings(projected_columns, columns);
    ArrowArrayStream stream{};
    impl_->open_stream(std::move(columns), reinterpret_cast<uint8_t*>(&stream));
    data_split_stream_open_count.fetch_add(1, std::memory_order_relaxed);
    return stream;
  } catch (const rust::cxxbridge1::Error& error) {
    throw PaimonException(error.what());
  }
}

}  // namespace milvus_storage::paimon
