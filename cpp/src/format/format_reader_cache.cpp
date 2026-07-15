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
  bool owns_in_flight_load = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    // Return cached immutable metadata without entering the singleflight path.
    auto cached = entries_.find(key);
    if (cached != entries_.end()) {
      return cached->second.metadata;
    }

    // The first caller for a key becomes the sync leader. A sync follower may
    // wait for another sync leader because that leader is actively loading on
    // a different thread. It must not block behind an async leader: doing so can
    // occupy the executor thread required to finish that async load.
    auto [it, inserted] = in_flight_loads_.try_emplace(key, std::make_shared<InFlightLoad>(InFlightLoad::kSync));
    in_flight_load = it->second;
    owns_in_flight_load = inserted;
    if (!inserted && in_flight_load->leader_type == InFlightLoad::kSync) {
      // condition_variable::wait releases mutex_ while sleeping, so other keys
      // and the leader can still update the cache. The current thread remains blocked.
      in_flight_load->cv.wait(lock, [&in_flight_load]() { return in_flight_load->done; });
      if (!in_flight_load->status.ok()) {
        return in_flight_load->status;
      }
      return in_flight_load->metadata;
    }

    // If the existing leader is async, fall through and load independently on
    // this thread. This intentionally trades a rare duplicate metadata read for
    // executor-independent progress of the synchronous API.
  }

  // Run the loader outside mutex_ so unrelated cache operations are not serialized
  // behind metadata I/O. The leader publishes both success and failure below.
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

  // The sync leader owns the InFlightLoad and must publish its result to all
  // followers. An independent sync load must leave the async leader's marker
  // and promise untouched, otherwise that leader could complete them twice.
  if (owns_in_flight_load) {
    if (!status.ok()) {
      return complete_load(key, in_flight_load, status);
    }
    return complete_load(key, in_flight_load, std::move(metadata));
  }

  if (!status.ok()) {
    // This bypass is intentionally independent of the async flight. Return its
    // own load error without waiting for or completing the async leader.
    return status;
  }

  // The async leader may have populated the cache while this independent load
  // was running. Preserve the first successful cached value so both paths
  // converge on one immutable metadata instance when both loads succeed.
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.try_emplace(key, Entry{std::move(metadata)}).first;
  return it->second.metadata;
}

template <typename ReaderT>
folly::SemiFuture<typename FormatReaderMetadataCache<ReaderT>::MetadataResult>
FormatReaderMetadataCache<ReaderT>::get_or_open_async(
    const std::string& key, const typename FormatReaderMetadataCache<ReaderT>::AsyncMetadataLoader& load_fn) {
  // Keep the cache alive for the whole deferred operation. Lookup and loading do
  // not start until the returned SemiFuture is consumed.
  auto self = this->shared_from_this();

  return folly::makeSemiFuture().deferValue([self = std::move(self), key,
                                             load_fn](folly::Unit) -> folly::SemiFuture<MetadataResult> {
    std::shared_ptr<InFlightLoad> in_flight_load;
    {
      std::lock_guard<std::mutex> lock(self->mutex_);

      // Recheck the cache at execution time because another caller may have
      // populated it after get_or_open_async() returned its deferred future.
      auto cached = self->entries_.find(key);
      if (cached != self->entries_.end()) {
        return folly::makeSemiFuture(MetadataResult(cached->second.metadata));
      }

      // Async followers subscribe to the leader's SharedPromise instead of
      // blocking a thread. Only the caller that inserts the marker runs load_fn.
      auto [it, inserted] = self->in_flight_loads_.try_emplace(key, std::make_shared<InFlightLoad>());
      in_flight_load = it->second;
      if (!inserted) {
        return in_flight_load->async_result.getSemiFuture();
      }
    }

    // Start the async loader outside mutex_. Its continuation normalizes and
    // publishes the result through the same path used by the synchronous leader.
    try {
      return load_fn().defer([self, key, in_flight_load](folly::Try<MetadataResult>&& load_try) -> MetadataResult {
        if (load_try.hasException()) {
          auto message = load_try.exception().what();
          return self->complete_load(
              key, in_flight_load,
              arrow::Status::UnknownError("Exception while asynchronously loading format reader metadata: ",
                                          std::string(message.data(), message.size())));
        }
        return self->complete_load(key, in_flight_load, std::move(load_try).value());
      });
    } catch (const std::exception& e) {
      return folly::makeSemiFuture(self->complete_load(
          key, in_flight_load,
          arrow::Status::UnknownError("Exception while asynchronously loading format reader metadata: ", e.what())));
    } catch (...) {
      return folly::makeSemiFuture(self->complete_load(
          key, in_flight_load,
          arrow::Status::UnknownError("Unknown exception while asynchronously loading format reader metadata")));
    }
  });
}

template <typename ReaderT>
typename FormatReaderMetadataCache<ReaderT>::MetadataResult FormatReaderMetadataCache<ReaderT>::complete_load(
    const std::string& key,
    const std::shared_ptr<InFlightLoad>& in_flight_load,
    typename FormatReaderMetadataCache<ReaderT>::MetadataResult load_result) {
  // Normalize a successful-but-null loader result into an error before exposing
  // it to the cache or any waiter.
  auto status = load_result.status();
  MetadataPtr metadata;
  if (load_result.ok()) {
    metadata = std::move(load_result).ValueOrDie();
    if (!metadata) {
      status = arrow::Status::Invalid("Format reader metadata loader returned null metadata");
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Atomically publish successful metadata and mark this singleflight as done.
    // A successful leader adopts an entry populated first by another successful
    // path. A failed leader still completes its own waiters with that failure.
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

    // Erase only this flight: a newer retry for the same key must not be removed
    // by a stale completion.
    auto in_flight_it = in_flight_loads_.find(key);
    if (in_flight_it != in_flight_loads_.end() && in_flight_it->second == in_flight_load) {
      in_flight_loads_.erase(in_flight_it);
    }
  }

  // Notify blocking synchronous waiters only after releasing mutex_.
  in_flight_load->cv.notify_all();

  // Fulfill the shared result consumed by every asynchronous follower.
  if (status.ok()) {
    in_flight_load->async_result.setValue(MetadataResult(metadata));
  } else {
    in_flight_load->async_result.setValue(MetadataResult(status));
  }

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
