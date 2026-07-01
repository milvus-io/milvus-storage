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
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "milvus-storage/common/lru_memory_cache.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage {

constexpr uint64_t kDefaultMetadataCacheCapacityBytes = 1ULL * 1024 * 1024 * 1024;

using GlobalFormatReaderMetadataCaches = FormatReaderMetadataCaches<parquet::ParquetFormatReader,
                                                                    vortex::VortexFormatReader,
                                                                    lance::LanceTableReader,
                                                                    iceberg::IcebergFormatReader>;

using GlobalCompletedMetadata = std::variant<FormatReaderMetadataCache<parquet::ParquetFormatReader>::MetadataPtr,
                                             FormatReaderMetadataCache<vortex::VortexFormatReader>::MetadataPtr,
                                             FormatReaderMetadataCache<lance::LanceTableReader>::MetadataPtr,
                                             FormatReaderMetadataCache<iceberg::IcebergFormatReader>::MetadataPtr>;

LRUMemoryCache<std::string, GlobalCompletedMetadata>& GlobalCompletedMetadataCache() {
  static LRUMemoryCache<std::string, GlobalCompletedMetadata> cache(kDefaultMetadataCacheCapacityBytes);
  return cache;
}

GlobalFormatReaderMetadataCaches& GlobalTypedMetadataCaches() {
  static GlobalFormatReaderMetadataCaches caches;
  return caches;
}

template <typename ReaderT>
std::optional<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> GetGlobalMetadata(const std::string& key) {
  auto cached = GlobalCompletedMetadataCache().get(key);
  if (!cached.has_value()) {
    return std::nullopt;
  }

  using MetadataPtr = typename FormatReaderMetadataCache<ReaderT>::MetadataPtr;
  auto* metadata = std::get_if<MetadataPtr>(&cached.value());
  if (metadata == nullptr) {
    return std::nullopt;
  }
  return *metadata;
}

template <typename ReaderT>
bool PutGlobalMetadata(const std::string& key,
                       typename FormatReaderMetadataCache<ReaderT>::MetadataPtr metadata,
                       uint64_t size_bytes) {
  return GlobalCompletedMetadataCache().put(key, GlobalCompletedMetadata{std::move(metadata)}, size_bytes);
}

void InitMetadataCache(uint64_t capacity_bytes) { GlobalCompletedMetadataCache().set_capacity(capacity_bytes); }

void ClearMetadataCache() { GlobalCompletedMetadataCache().clear(); }

arrow::Result<std::string> MakeFormatReaderMetadataCacheKey(const api::ColumnGroupFile& file,
                                                            const api::Properties& properties,
                                                            const std::string& format_cache_key) {
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, file.path));
  return format_cache_key + "|fs_cache_key=" + fs_config.GetCacheKey();
}

template <typename ReaderT>
std::shared_ptr<FormatReaderMetadataCache<ReaderT>> GetGlobalFormatReaderMetadataCache() {
  static_assert(FormatReaderWithMetadata<ReaderT>,
                "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                "cache_key, load_metadata, and create_from_metadata.");
  return GlobalTypedMetadataCaches().get<ReaderT>();
}

template <typename ReaderT>
std::optional<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> FormatReaderMetadataCache<ReaderT>::get(
    const std::string& key) const {
  return GetGlobalMetadata<ReaderT>(key);
}

template <typename ReaderT>
arrow::Status FormatReaderMetadataCache<ReaderT>::add(
    std::string key, typename FormatReaderMetadataCache<ReaderT>::MetadataPtr metadata) {
  if (!metadata) {
    return arrow::Status::Invalid("Cannot add null format reader metadata to cache");
  }

  const auto size_bytes = metadata->cache_size;
  PutGlobalMetadata<ReaderT>(key, std::move(metadata), size_bytes);
  return arrow::Status::OK();
}

template <typename ReaderT>
arrow::Result<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> FormatReaderMetadataCache<ReaderT>::get_or_open(
    const std::string& key, const typename FormatReaderMetadataCache<ReaderT>::MetadataLoader& load_fn) {
  std::shared_ptr<InFlightLoad> in_flight_load;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto cached = GetGlobalMetadata<ReaderT>(key);
    if (cached.has_value()) {
      return cached.value();
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
      auto cached = GetGlobalMetadata<ReaderT>(key);
      if (cached.has_value()) {
        metadata = cached.value();
      } else if (metadata->cache_size != 0) {
        PutGlobalMetadata<ReaderT>(key, metadata, metadata->cache_size);
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

template std::shared_ptr<FormatReaderMetadataCache<parquet::ParquetFormatReader>>
GetGlobalFormatReaderMetadataCache<parquet::ParquetFormatReader>();
template std::shared_ptr<FormatReaderMetadataCache<vortex::VortexFormatReader>>
GetGlobalFormatReaderMetadataCache<vortex::VortexFormatReader>();
template std::shared_ptr<FormatReaderMetadataCache<lance::LanceTableReader>>
GetGlobalFormatReaderMetadataCache<lance::LanceTableReader>();
template std::shared_ptr<FormatReaderMetadataCache<iceberg::IcebergFormatReader>>
GetGlobalFormatReaderMetadataCache<iceberg::IcebergFormatReader>();

template class FormatReaderMetadataCache<parquet::ParquetFormatReader>;
template class FormatReaderMetadataCache<vortex::VortexFormatReader>;
template class FormatReaderMetadataCache<lance::LanceTableReader>;
template class FormatReaderMetadataCache<iceberg::IcebergFormatReader>;

}  // namespace milvus_storage
