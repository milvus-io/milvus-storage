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

#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/properties.h"

namespace milvus_storage {

// Configure the process-wide completed metadata cache byte capacity. Setting
// capacity to 0 evicts existing entries and prevents future retention, while
// loaders still return metadata to their callers.
void InitMetadataCache(uint64_t capacity_bytes);

// Clear completed metadata entries from the process-wide cache. This is not a
// barrier for in-flight loads; a load already running may still complete and
// populate the cache after this call returns.
void ClearMetadataCache();

arrow::Result<std::string> MakeFormatReaderMetadataCacheKey(const api::ColumnGroupFile& file,
                                                            const api::Properties& properties,
                                                            const std::string& format_cache_key);

// Thread-safe metadata cache for one concrete FormatReader type.
// Cached metadata is immutable and can be reused to create independent
// stateful readers with different projections or predicates. Completed entries
// are stored process-wide, but each direct FormatReaderMetadataCache instance
// owns its own singleflight state. Production reader code should obtain typed
// caches through GetGlobalFormatReaderMetadataCache<ReaderT>() when process-wide
// singleflight is required.
template <typename ReaderT>
class FormatReaderMetadataCache final {
  static_assert(FormatReaderWithMetadata<ReaderT>,
                "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                "cache_key, load_metadata, and create_from_metadata.");

  public:
  using ReaderType = ReaderT;
  using Trait = typename FormatReader::template MetaTrait<ReaderT>;
  using MetadataPtr = typename Trait::MetadataPtr;
  using MetadataLoader = std::function<arrow::Result<MetadataPtr>()>;

  std::optional<MetadataPtr> get(const std::string& key) const;

  // Adds metadata to the process-wide completed-entry cache. Null metadata is
  // rejected. Zero-size metadata, and metadata larger than the current global
  // capacity, is accepted but not retained.
  arrow::Status add(std::string key, MetadataPtr metadata);

  arrow::Result<MetadataPtr> get_or_open(const std::string& key, const MetadataLoader& load_fn);

  private:
  // Per-key singleflight state. The first cache miss creates this marker and
  // runs load_fn outside mutex_; waiters for the same key block on cv while
  // unrelated keys can still load concurrently.
  struct InFlightLoad {
    std::condition_variable cv;
    bool done = false;
    arrow::Status status = arrow::Status::OK();
    MetadataPtr metadata;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<InFlightLoad>> in_flight_loads_;
};

// Owns one typed metadata cache for each ReaderT in the template list.
// Callers still retrieve caches statically with get<ReaderT>(); this class only
// groups the per-format caches into one value that can be embedded elsewhere.
template <typename... ReaderTs>
class FormatReaderMetadataCaches final {
  public:
  FormatReaderMetadataCaches() : caches_(std::make_shared<FormatReaderMetadataCache<ReaderTs>>()...) {}

  template <typename ReaderT>
  [[nodiscard]] std::shared_ptr<FormatReaderMetadataCache<ReaderT>> get() const {
    static_assert((std::same_as<ReaderT, ReaderTs> || ...), "ReaderT must be a supported metadata cache reader type");
    return std::get<std::shared_ptr<FormatReaderMetadataCache<ReaderT>>>(caches_);
  }

  private:
  std::tuple<std::shared_ptr<FormatReaderMetadataCache<ReaderTs>>...> caches_;
};

template <typename ReaderT>
std::shared_ptr<FormatReaderMetadataCache<ReaderT>> GetGlobalFormatReaderMetadataCache();

}  // namespace milvus_storage
