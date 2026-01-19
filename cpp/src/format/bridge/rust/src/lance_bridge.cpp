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

std::shared_ptr<BlockingDataset> BlockingDataset::Open(const std::string& uri) {
  try {
    return std::make_shared<BlockingDataset>(ffi::open_dataset(rust::Str(uri.data(), uri.length())));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingDataset> BlockingDataset::OpenUnique(const std::string& uri) {
  try {
    return std::make_unique<BlockingDataset>(ffi::open_dataset(rust::Str(uri.data(), uri.length())));
  } catch (const rust::cxxbridge1::Error& e) {
    throw LanceException(e.what());
  }
}

std::unique_ptr<BlockingDataset> BlockingDataset::WriteDataset(const std::string& uri,
                                                               struct ArrowArrayStream* stream) {
  try {
    return std::make_unique<BlockingDataset>(
        ffi::write_dataset(rust::Str(uri.data(), uri.length()), reinterpret_cast<uint8_t*>(stream)));
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

}  // namespace milvus_storage::lance
