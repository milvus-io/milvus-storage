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

#include "milvus-storage/format/lance/lance_table_reader.h"

#include <atomic>
#include <compare>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <utility>

#include <folly/Indestructible.h>
#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <folly/futures/SharedPromise.h>
#include <glog/raw_logging.h>

#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/format/lance/lance_common.h"

namespace milvus_storage::lance {
namespace {

// Lance metadata follows the Dataset/fragment hierarchy and uses three cache
// levels with distinct identities and ownership:
//
//   LanceDatasetCache (process-wide)
//     key: {dataset version, base URI, filesystem cache key}
//     value: weak_ptr<BlockingDataset>
//
//   FormatReaderMetadataCache (owned by one top-level Reader)
//     key: base URI
//     value: Metadata -> Payload -> shared_ptr<BlockingDataset>
//
//   Payload::FragmentMetadataCache
//     key: fragment ID
//     value: immutable fragment schema, row groups, deletion state, and
//            memory estimates
//
// LanceFormat::explore() records the already-open Dataset's version in each
// ColumnGroupFile. A reader can therefore query LanceDatasetCache before the
// expensive Dataset open. Legacy files without that property resolve only the
// latest manifest location first. Exact-version singleflight ensures concurrent
// misses decode and retain one Dataset snapshot. The process cache owns only
// weak pointers, so top-level reader metadata determines Dataset lifetime.
//
//   LanceDatasetCache
//     `-- weak_ptr<BlockingDataset> -------------------------+
//                                                            |
//   ReaderImpl                                               |
//     `-- MetadataCache                                      |
//           `-- FormatReaderMetadataCache<LanceTableReader>  |
//                 `-- Metadata -> Payload                    |
//                       +-- shared_ptr<BlockingDataset> ------+
//                       `-- FragmentMetadataCache
//                             +-- fragment 0 -> metadata[0]
//                             `-- fragment 1 -> metadata[1]
//
// BlockingFragmentReader is projection-specific and stateful, so every
// LanceTableReader creates its own instance rather than caching it.
class LanceDatasetCache final {
  public:
  using DatasetPtr = std::shared_ptr<BlockingDataset>;

  struct Key {
    uint64_t version;
    std::string base_uri;
    std::string filesystem_cache_key;

    bool operator==(const Key&) const = default;
    auto operator<=>(const Key&) const = default;
  };

  static LanceDatasetCache& Instance() {
    // Rust runtime tasks can finish during C++ static teardown. Their callbacks
    // must still be able to access this process-wide cache.
    static folly::Indestructible<LanceDatasetCache> cache;
    return *cache;
  }

  template <typename DatasetLoader>
  arrow::Result<DatasetPtr> GetOrOpen(const Key& key, DatasetLoader&& load_fn) {
    // 1. Reuse a cached dataset or wait for an existing sync open.
    std::shared_ptr<InFlightOpen> in_flight_open;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      auto cached = datasets_.get(key);
      if (cached.has_value()) {
        if (auto dataset = cached->lock()) {
          return dataset;
        }
        datasets_.remove(key);
      }

      const auto existing_open = in_flight_opens_.find(key);
      if (existing_open != in_flight_opens_.end()) {
        in_flight_open = existing_open->second;
        // As in FormatReaderMetadataCache, a synchronous caller must not wait
        // for an async leader that may need this caller's executor to finish.
        if (!in_flight_open->async_leader) {
          in_flight_open->cv.wait(lock, [&in_flight_open]() { return in_flight_open->done; });
          if (!in_flight_open->status.ok()) {
            return in_flight_open->status;
          }
          return in_flight_open->dataset;
        }
      } else {
        in_flight_open = std::make_shared<InFlightOpen>();
        in_flight_opens_.emplace(key, in_flight_open);
      }
    }

    // 2. Open outside the lock without waiting for an async leader.
    auto status = arrow::Status::OK();
    DatasetPtr dataset;
    try {
      auto load_result = load_fn();
      status = load_result.status();
      if (load_result.ok()) {
        dataset = std::move(load_result).ValueOrDie();
        if (!dataset) {
          status = arrow::Status::Invalid("Lance dataset loader returned null for base URI: ", key.base_uri,
                                          ", version: ", key.version);
        }
      }
    } catch (const std::exception& e) {
      status = arrow::Status::UnknownError("Exception while opening Lance dataset for base URI ", key.base_uri,
                                           ", version ", key.version, ": ", e.what());
    } catch (...) {
      status = arrow::Status::UnknownError("Unknown exception while opening Lance dataset for base URI: ", key.base_uri,
                                           ", version: ", key.version);
    }

    // 3. Cache the result; finish the flight only if this call started it.
    if (in_flight_open->async_leader) {
      if (!status.ok()) {
        return status;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      auto cached = datasets_.get(key);
      if (cached.has_value()) {
        if (auto existing = cached->lock()) {
          return existing;
        }
      }
      datasets_.put(key, std::weak_ptr<BlockingDataset>(dataset));
      return dataset;
    }
    return CompleteOpen(key, in_flight_open, status.ok() ? arrow::Result<DatasetPtr>(dataset) : status);
  }

  template <typename DatasetLoader>
  folly::SemiFuture<arrow::Result<DatasetPtr>> GetOrOpenAsync(const Key& key, DatasetLoader load_fn) {
    return folly::makeSemiFuture().deferValue(
        [this, key, load_fn = std::move(load_fn)](folly::Unit) -> folly::SemiFuture<arrow::Result<DatasetPtr>> {
          // 1. Reuse a cached dataset or subscribe to an existing open.
          std::shared_ptr<InFlightOpen> flight;
          {
            std::lock_guard<std::mutex> lock(mutex_);
            auto cached = datasets_.get(key);
            if (cached.has_value()) {
              if (auto dataset = cached->lock()) {
                return folly::makeSemiFuture(arrow::Result<DatasetPtr>(std::move(dataset)));
              }
              datasets_.remove(key);
            }
            auto [it, inserted] = in_flight_opens_.try_emplace(key, std::make_shared<InFlightOpen>());
            flight = it->second;
            if (!inserted) {
              return flight->async_result.getSemiFuture();
            }
            flight->async_leader = true;
          }
          // 2. Start the async open outside the lock, with cleanup if abandoned.
          try {
            auto future = load_fn();
            // A timed SemiFuture::get() can discard the deferred continuation.
            // Release its singleflight marker even when completion is abandoned.
            auto abandoned = folly::makeGuard([this, key, flight]() noexcept {
              if (!flight->continuation_attached.load(std::memory_order_acquire)) {
                return;
              }
              auto cleanup = folly::makeTryWith([&] {
                return CompleteOpen(key, flight, arrow::Status::Cancelled("Lance dataset open was abandoned"));
              });
              if (cleanup.hasException()) {
                const auto* error = cleanup.exception().get_exception();
                RAW_LOG(ERROR, "Failed to abandon Lance dataset open: %s", error ? error->what() : "unknown exception");
              }
            });
            // 3. Publish the result and notify all waiters.
            auto pending = std::move(future).defer([this, key, flight, abandoned = std::move(abandoned)](
                                                       folly::Try<arrow::Result<DatasetPtr>>&& result) mutable {
              auto loaded = result.hasException() ? arrow::Result<DatasetPtr>(arrow::Status::UnknownError(
                                                        "Exception while asynchronously opening Lance dataset: ",
                                                        result.exception().what().toStdString()))
                                                  : std::move(result).value();
              auto completed = CompleteOpen(key, flight, std::move(loaded));
              abandoned.dismiss();
              return completed;
            });
            // Attachment failures belong to the catch below, not to cancellation.
            flight->continuation_attached.store(true, std::memory_order_release);
            return pending;
          } catch (const std::exception& error) {
            return folly::makeSemiFuture(CompleteOpen(
                key, flight, arrow::Status::UnknownError("Failed to submit Lance dataset open: ", error.what())));
          } catch (...) {
            return folly::makeSemiFuture(CompleteOpen(
                key, flight, arrow::Status::UnknownError("Unknown exception submitting Lance dataset open")));
          }
        });
  }

  private:
  struct InFlightOpen {
    bool done = false;
    bool async_leader = false;
    std::atomic<bool> continuation_attached{false};
    arrow::Status status = arrow::Status::OK();
    DatasetPtr dataset;
    std::condition_variable cv;
    folly::SharedPromise<arrow::Result<DatasetPtr>> async_result;
  };

  arrow::Result<DatasetPtr> CompleteOpen(const Key& key,
                                         const std::shared_ptr<InFlightOpen>& in_flight_open,
                                         arrow::Result<DatasetPtr> result) {
    auto status = result.status();
    auto dataset = result.ok() ? std::move(result).ValueOrDie() : nullptr;
    if (status.ok() && !dataset) {
      status = arrow::Status::Invalid("Lance dataset loader returned null for base URI: ", key.base_uri);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (in_flight_open->done) {
        return in_flight_open->status.ok() ? arrow::Result<DatasetPtr>(in_flight_open->dataset)
                                           : in_flight_open->status;
      }
      if (status.ok()) {
        auto cached = datasets_.get(key);
        if (cached.has_value()) {
          if (auto existing = cached->lock()) {
            dataset = std::move(existing);
          }
        }
        try {
          datasets_.put(key, std::weak_ptr<BlockingDataset>(dataset));
        } catch (...) {
          // Publication is best effort. The opened dataset must still reach all
          // waiters so an allocation failure cannot strand the in-flight open.
        }
      }
      in_flight_open->status = status;
      in_flight_open->dataset = dataset;
      in_flight_open->done = true;

      auto in_flight_it = in_flight_opens_.find(key);
      if (in_flight_it != in_flight_opens_.end() && in_flight_it->second == in_flight_open) {
        in_flight_opens_.erase(in_flight_it);
      }
    }
    in_flight_open->cv.notify_all();
    arrow::Result<DatasetPtr> completed = status.ok() ? arrow::Result<DatasetPtr>(dataset) : status;
    in_flight_open->async_result.setValue(completed);
    return completed;
  }

  // Weak entries are cheap, while this larger bound absorbs normal snapshot
  // churn without allowing versioned keys to grow for the process lifetime.
  static constexpr size_t kDatasetCacheCapacity = 4096;

  std::mutex mutex_;
  LRUCache<Key, std::weak_ptr<BlockingDataset>> datasets_{kDatasetCacheCapacity};
  std::map<Key, std::shared_ptr<InFlightOpen>> in_flight_opens_;
};

}  // namespace

arrow::Result<std::shared_ptr<BlockingDataset>> LanceTableReader::get_or_open_dataset(
    const std::string& base_uri,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const StorageOptions& read_options,
    const std::string& filesystem_cache_key,
    uint64_t version) {
  const LanceDatasetCache::Key key{version, base_uri, filesystem_cache_key};
  return LanceDatasetCache::Instance().GetOrOpen(
      key, [&]() { return BlockingDataset::Open(ToStandardLanceUri(base_uri), filesystem, read_options, version); });
}

folly::SemiFuture<arrow::Result<std::shared_ptr<BlockingDataset>>> LanceTableReader::get_or_open_dataset_async(
    const std::string& base_uri,
    const std::shared_ptr<arrow::fs::FileSystem>& filesystem,
    const StorageOptions& read_options,
    const std::string& filesystem_cache_key,
    uint64_t version) {
  const LanceDatasetCache::Key key{version, base_uri, filesystem_cache_key};
  return LanceDatasetCache::Instance().GetOrOpenAsync(key, [base_uri, filesystem, read_options, version]() {
    return load_dataset_async(ToStandardLanceUri(base_uri), filesystem, read_options, version);
  });
}

}  // namespace milvus_storage::lance
