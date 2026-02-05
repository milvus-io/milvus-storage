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

namespace milvus_storage::lance {

void ReplaceLanceRuntime(uint32_t num_threads) {}

namespace {
// Helper to convert LanceStorageOptions to rust::Vec pairs for FFI
// Note: rust::String(std::string const&) constructor is not available on Linux,
// so we use rust::String(data, length) instead.
void ConvertStorageOptions(const LanceStorageOptions& storage_options,
                           rust::Vec<rust::String>& keys,
                           rust::Vec<rust::String>& values) {
  for (const auto& [k, v] : storage_options) {
    keys.push_back(rust::String(k.data(), k.length()));
    values.push_back(rust::String(v.data(), v.length()));
  }
}
}  // namespace

std::shared_ptr<BlockingDataset> BlockingDataset::Open(const std::string& uri,
                                                       const LanceStorageOptions& storage_options) {
  try {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    return std::make_shared<BlockingDataset>(
        ffi::open_dataset(rust::Str(uri.data(), uri.length()), std::move(keys), std::move(values)));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingDataset> BlockingDataset::OpenUnique(const std::string& uri,
                                                             const LanceStorageOptions& storage_options) {
  try {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    return std::make_unique<BlockingDataset>(
        ffi::open_dataset(rust::Str(uri.data(), uri.length()), std::move(keys), std::move(values)));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingDataset> BlockingDataset::WriteDataset(const std::string& uri,
                                                               struct ArrowArrayStream* stream,
                                                               const LanceStorageOptions& storage_options,
                                                               LanceDataStorageFormat format) {
  try {
    rust::Vec<rust::String> keys, values;
    ConvertStorageOptions(storage_options, keys, values);
    auto ffi_format = static_cast<ffi::LanceDataStorageFormat>(format);
    return std::make_unique<BlockingDataset>(ffi::write_dataset(rust::Str(uri.data(), uri.length()),
                                                                reinterpret_cast<uint8_t*>(stream), std::move(keys),
                                                                std::move(values), ffi_format));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::vector<uint64_t> BlockingDataset::GetAllFragmentIds() const {
  try {
    auto fragment_ids = impl_->get_all_fragment_ids();
    return std::vector<uint64_t>(fragment_ids.begin(), fragment_ids.end());
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

void BlockingDataset::WriteArrowArrayStream(struct ArrowArrayStream* stream) {
  try {
    impl_->write_stream(reinterpret_cast<uint8_t*>(stream));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingFragmentReader> BlockingFragmentReader::Open(const BlockingDataset& dataset,
                                                                     uint64_t fragment_id,
                                                                     ArrowSchema& schema) {
  try {
    auto impl = ffi::open_fragment_reader(dataset.Impl(), fragment_id, reinterpret_cast<uint8_t*>(&schema));
    return std::make_unique<BlockingFragmentReader>(std::move(impl));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

uint64_t BlockingFragmentReader::RowCount() const {
  try {
    return impl_->number_of_rows();
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

void BlockingFragmentReader::TakeAsSingleBatch(const std::vector<int64_t>& indices, ArrowArray& out_array) {
  try {
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_single_batch(indices_slice, reinterpret_cast<uint8_t*>(&out_array));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

ArrowArrayStream BlockingFragmentReader::TakeAsStream(const std::vector<int64_t>& indices, uint32_t batch_size) {
  try {
    ArrowArrayStream stream;
    std::vector<uint32_t> uint32_indices(indices.begin(), indices.end());
    rust::Slice<const uint32_t> indices_slice(uint32_indices.data(), uint32_indices.size());
    impl_->take_as_stream(indices_slice, batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

ArrowArrayStream BlockingFragmentReader::ReadAllAsStream(uint32_t batch_size) {
  try {
    ArrowArrayStream stream;
    impl_->read_all_as_stream(batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

ArrowArrayStream BlockingFragmentReader::ReadRangesAsStream(uint32_t row_range_start,
                                                            uint32_t row_range_end,
                                                            uint32_t batch_size) {
  try {
    ArrowArrayStream stream;
    impl_->read_ranges_as_stream(row_range_start, row_range_end, batch_size, reinterpret_cast<uint8_t*>(&stream));
    return stream;
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingScanner> BlockingDataset::Scan(ArrowSchema& schema, uint32_t batch_size) {
  try {
    auto impl = ffi::create_scanner(*impl_, reinterpret_cast<uint8_t*>(&schema), batch_size);
    return std::make_unique<BlockingScanner>(std::move(impl));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

ArrowArrayStream BlockingDataset::Take(const std::vector<int64_t>& indices, ArrowSchema& schema) {
  try {
    ArrowArrayStream stream;
    std::vector<uint64_t> uint64_indices(indices.begin(), indices.end());
    rust::Slice<const uint64_t> indices_slice(uint64_indices.data(), uint64_indices.size());
    ffi::dataset_take(*impl_, indices_slice, reinterpret_cast<uint8_t*>(&schema), reinterpret_cast<uint8_t*>(&stream));
    return stream;
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

uint64_t BlockingScanner::CountRows() const {
  try {
    return impl_->count_rows();
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

ArrowArrayStream BlockingScanner::OpenStream() {
  try {
    ArrowArrayStream stream;
    impl_->open_stream(reinterpret_cast<uint8_t*>(&stream));
    return stream;
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

}  // namespace milvus_storage::lance
