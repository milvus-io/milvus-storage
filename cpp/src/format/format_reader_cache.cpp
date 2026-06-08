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

#include "milvus-storage/format/format_reader_cache.h"

#include <exception>
#include <utility>

#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage {

MetadataCache::MetadataCache(bool enabled) : enabled_(enabled) {}

template <typename ReaderT>
std::optional<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> FormatReaderMetadataCache<ReaderT>::get(
    const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second.metadata;
}

template <typename ReaderT>
arrow::Status FormatReaderMetadataCache<ReaderT>::add(
    std::string key, typename FormatReaderMetadataCache<ReaderT>::MetadataPtr metadata) {
  if (!metadata) {
    return arrow::Status::Invalid("Cannot add null format reader metadata to cache");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  entries_[std::move(key)] = Entry{std::move(metadata)};
  return arrow::Status::OK();
}

template <typename ReaderT>
arrow::Result<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> FormatReaderMetadataCache<ReaderT>::get_or_open(
    const std::string& key, const typename FormatReaderMetadataCache<ReaderT>::MetadataLoader& load_fn) {
  std::shared_ptr<InFlightLoad> in_flight_load;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto cached = entries_.find(key);
    if (cached != entries_.end()) {
      return cached->second.metadata;
    }

    auto [it, inserted] = in_flight_loads_.try_emplace(key, std::make_shared<InFlightLoad>());
    in_flight_load = it->second;
    if (!inserted) {
      in_flight_load->cv.wait(lock, [&in_flight_load]() { return in_flight_load->done; });
      if (!in_flight_load->status.ok()) {
        return in_flight_load->status;
      }
      return in_flight_load->metadata;
    }
  }

  auto status = arrow::Status::OK();
  MetadataPtr metadata;
  try {
    auto load_result = load_fn();
    status = load_result.status();
    if (load_result.ok()) {
      metadata = load_result.ValueOrDie();
      if (!metadata) {
        status = arrow::Status::Invalid("Format reader metadata loader returned null metadata");
      }
    }
  } catch (const std::exception& e) {
    status = arrow::Status::UnknownError("Exception while loading format reader metadata: ", e.what());
  } catch (...) {
    status = arrow::Status::UnknownError("Unknown exception while loading format reader metadata");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status.ok()) {
      auto cached = entries_.find(key);
      if (cached != entries_.end()) {
        metadata = cached->second.metadata;
      } else {
        entries_.emplace(key, Entry{metadata});
      }
    }

    in_flight_load->status = status;
    in_flight_load->metadata = metadata;
    in_flight_load->done = true;
    in_flight_loads_.erase(key);
  }
  in_flight_load->cv.notify_all();

  if (!status.ok()) {
    return status;
  }
  return metadata;
}

template class FormatReaderMetadataCache<parquet::ParquetFormatReader>;
template class FormatReaderMetadataCache<vortex::VortexFormatReader>;
template class FormatReaderMetadataCache<lance::LanceTableReader>;
template class FormatReaderMetadataCache<iceberg::IcebergFormatReader>;

}  // namespace milvus_storage
