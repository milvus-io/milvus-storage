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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <typeindex>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>

#include "milvus-storage/format/format_reader.h"

namespace milvus_storage {

namespace iceberg {
class IcebergFormatReader;
}  // namespace iceberg

namespace lance {
class LanceTableReader;
}  // namespace lance

namespace parquet {
class ParquetFormatReader;
}  // namespace parquet

namespace vortex {
class VortexFormatReader;
}  // namespace vortex

// Thread-safe metadata cache for one concrete FormatReader type.
// Cached metadata is immutable and can be reused to create independent
// stateful readers with different projections or predicates.
template <typename ReaderT>
class FormatReaderMetadataCache final : public std::enable_shared_from_this<FormatReaderMetadataCache<ReaderT>> {
  static_assert(FormatReaderWithMetadata<ReaderT>,
                "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                "cache_key, load_metadata, and create_from_metadata.");

  public:
  using ReaderType = ReaderT;
  using Trait = typename FormatReader::template MetaTrait<ReaderT>;
  using MetadataPtr = typename Trait::MetadataPtr;
  using MetadataResult = arrow::Result<MetadataPtr>;
  using MetadataLoader = std::function<arrow::Result<MetadataPtr>()>;
  using AsyncMetadataLoader = std::function<folly::SemiFuture<MetadataResult>()>;

  // Construct with shared ownership required by get_or_open_async().
  [[nodiscard]] static std::shared_ptr<FormatReaderMetadataCache> Make() {
    return std::shared_ptr<FormatReaderMetadataCache>(new FormatReaderMetadataCache());
  }

  std::optional<MetadataPtr> get(const std::string& key) const;

  arrow::Status add(std::string key, MetadataPtr metadata);

  arrow::Result<MetadataPtr> get_or_open(const std::string& key, const MetadataLoader& load_fn);

  // Coalesce same-key metadata loads and share the leader result with async followers.
  // The returned future retains the cache; failures are published but not cached.
  folly::SemiFuture<MetadataResult> get_or_open_async(const std::string& key, const AsyncMetadataLoader& load_fn);

  private:
  // Force callers through Make() so shared_from_this() is always valid.
  FormatReaderMetadataCache() = default;

  struct Entry {
    MetadataPtr metadata;
  };

  // Per-key singleflight state shared by synchronous and asynchronous callers.
  // Async followers always receive a future. Sync followers block only behind
  // sync leaders; waiting behind an async leader could starve its executor.
  struct InFlightLoad {
    enum LeaderType {
      kAsync,
      kSync,
    };

    // Record whether synchronous followers may safely block on this flight.
    explicit InFlightLoad(LeaderType leader_type = kAsync) : leader_type(leader_type) {}

    bool done = false;
    arrow::Status status = arrow::Status::OK();
    LeaderType leader_type;
    std::condition_variable cv;
    MetadataPtr metadata;
    folly::SharedPromise<MetadataResult> async_result;
  };

  // Publish the leader's load result to the cache and every same-key waiter.
  // Successful metadata is cached; failures are left uncached so a later call
  // can retry. A successful leader adopts metadata already cached by an
  // independent successful load; a failed leader still publishes its own error.
  // Both paths remove the in-flight load and complete that leader's waiters.
  MetadataResult complete_load(const std::string& key,
                               const std::shared_ptr<InFlightLoad>& in_flight_load,
                               MetadataResult load_result);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_map<std::string, std::shared_ptr<InFlightLoad>> in_flight_loads_;
};

// Owns one typed metadata cache for each ReaderT in the template list.
// Callers still retrieve caches statically with get<ReaderT>(); this class only
// groups the per-format caches into one value that can be embedded elsewhere.
template <typename... ReaderTs>
class FormatReaderMetadataCaches final {
  public:
  FormatReaderMetadataCaches() : caches_(FormatReaderMetadataCache<ReaderTs>::Make()...) {}

  template <typename ReaderT>
  [[nodiscard]] std::shared_ptr<FormatReaderMetadataCache<ReaderT>> get() const {
    static_assert((std::same_as<ReaderT, ReaderTs> || ...), "ReaderT must be a supported metadata cache reader type");
    return std::get<std::shared_ptr<FormatReaderMetadataCache<ReaderT>>>(caches_);
  }

  private:
  std::tuple<std::shared_ptr<FormatReaderMetadataCache<ReaderTs>>...> caches_;
};

// Public cache handle carried by ReaderImpl and passed down to column-group
// readers. Concrete reader headers are intentionally not included here, so
// installed consumers can include public reader headers without private bridge
// headers from the source tree.
class MetadataCache final {
  public:
  explicit MetadataCache(bool enabled = true);

  [[nodiscard]] bool enabled() const { return enabled_; }

  template <typename ReaderT>
  [[nodiscard]] std::shared_ptr<FormatReaderMetadataCache<ReaderT>> get() const {
    static_assert(FormatReaderWithMetadata<ReaderT>,
                  "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                  "cache_key, load_metadata, and create_from_metadata.");

    std::lock_guard<std::mutex> lock(state_->mutex);
    auto [it, inserted] = state_->caches.try_emplace(std::type_index(typeid(ReaderT)));
    if (inserted || !it->second) {
      it->second = FormatReaderMetadataCache<ReaderT>::Make();
    }
    return std::static_pointer_cast<FormatReaderMetadataCache<ReaderT>>(it->second);
  }

  template <typename Visitor>
  auto dispatch(const std::string& format, Visitor&& visitor) const {
    using ReturnT = decltype(std::forward<Visitor>(visitor)(get<parquet::ParquetFormatReader>()));

    if (format == LOON_FORMAT_PARQUET) {
      return std::forward<Visitor>(visitor)(get<parquet::ParquetFormatReader>());
    }
    if (format == LOON_FORMAT_VORTEX) {
      return std::forward<Visitor>(visitor)(get<vortex::VortexFormatReader>());
    }
    if (format == LOON_FORMAT_LANCE_TABLE) {
      return std::forward<Visitor>(visitor)(get<lance::LanceTableReader>());
    }
    if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return std::forward<Visitor>(visitor)(get<iceberg::IcebergFormatReader>());
    }

    return ReturnT(arrow::Status::Invalid("Unknown column group format: ", format));
  }

  private:
  struct State {
    mutable std::mutex mutex;
    std::unordered_map<std::type_index, std::shared_ptr<void>> caches;
  };

  bool enabled_ = true;
  std::shared_ptr<State> state_ = std::make_shared<State>();
};

}  // namespace milvus_storage
