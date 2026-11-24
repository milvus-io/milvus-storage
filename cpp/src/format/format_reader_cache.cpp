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

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

#include <folly/ScopeGuard.h>
#include <folly/Try.h>
#include <glog/raw_logging.h>

#include "cachinglayer/Manager.h"
#include "cachinglayer/CacheSlot.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/paimon/paimon_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage {

MetadataCache::MetadataCache(bool enabled) : enabled_(enabled) {}

template <typename ReaderT>
FormatReaderMetadataCache<ReaderT>::FormatReaderMetadataCache()
    : cache_(milvus::cachinglayer::Manager::GetInstance().CreateCacheMap<std::string, MetadataPtr>()) {}

template <typename ReaderT>
typename FormatReaderMetadataCache<ReaderT>::MetadataResult FormatReaderMetadataCache<ReaderT>::FromPin(
    typename KVCache::PinResult result) {
  if (!result) {
    return ExtendStatusDetail::WrapFromSegcoreError(result.error());
  }
  auto pin = std::move(result).value();
  if (!pin) {
    return MetadataPtr{};
  }
  auto metadata = *pin;
  try {
    // Keep the existing MetadataPtr API while tying every copy to the cache Pin.
    auto owner = std::make_shared<typename KVCache::Pin>(std::move(pin));
    auto lease = std::make_shared<milvus::cachinglayer::PinWrapper<MetadataPtr>>(std::move(owner), std::move(metadata));
    return MetadataPtr(std::move(lease), lease->get().get());
  } catch (const std::bad_alloc& error) {
    return arrow::Status::OutOfMemory(error.what());
  }
}

template <typename ReaderT>
typename FormatReaderMetadataCache<ReaderT>::MetadataResult FormatReaderMetadataCache<ReaderT>::get(
    const std::string& key) const {
  return FromPin(cache_->Get(key));
}

template <typename ReaderT>
arrow::Result<typename FormatReaderMetadataCache<ReaderT>::ResourceUsage>
FormatReaderMetadataCache<ReaderT>::reserve_for_load(const uint64_t estimated_bytes) const {
  if (estimated_bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return arrow::Status::CapacityError("Format reader metadata estimate exceeds int64");
  }
  const ResourceUsage reservation{static_cast<int64_t>(estimated_bytes), 0};
  if (estimated_bytes > 0) {
    const auto admission = cache_->Reserve(reservation);
    if (!admission) {
      return ExtendStatusDetail::WrapFromSegcoreError(admission.error());
    }
  }
  return reservation;
}

template <typename ReaderT>
void FormatReaderMetadataCache<ReaderT>::release_load_reservation(const ResourceUsage reservation) const {
  if (reservation.memory_bytes > 0) {
    cache_->UnReserve(reservation);
  }
}

template <typename ReaderT>
typename FormatReaderMetadataCache<ReaderT>::MetadataResult FormatReaderMetadataCache<ReaderT>::publish(
    MetadataPtr metadata, const ResourceUsage load_reservation) {
  if (!metadata || metadata->cache_key.empty()) {
    return arrow::Status::Invalid("Format reader metadata requires a value and a cache key");
  }
  const auto& key = metadata->cache_key;
  const auto bytes = std::max<uint64_t>(metadata->cache_size, sizeof(typename Trait::Metadata));
  if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return arrow::Status::CapacityError("Format reader metadata size exceeds int64");
  }
  const ResourceUsage usage{static_cast<int64_t>(bytes), 0};
  const ResourceUsage extra{std::max<int64_t>(0, usage.memory_bytes - load_reservation.memory_bytes), 0};
  bool reserved = false;
  auto refund = folly::makeGuard([&] {
    if (reserved) {
      cache_->UnReserve(extra);
    }
  });
  // Only a new entry needs any remaining bytes. A concurrent winner is pinned directly.
  return FromPin(cache_->GetOrCreate(key, [&]() -> typename KVCache::CreateResult {
    if (extra.memory_bytes > 0) {
      const auto admission = cache_->Reserve(extra);
      if (!admission) {
        return folly::makeUnexpected(admission.error());
      }
      reserved = true;
    }
    return std::pair{std::make_unique<MetadataPtr>(std::move(metadata)), usage};
  }));
}

template <typename ReaderT>
arrow::Status FormatReaderMetadataCache<ReaderT>::add(MetadataPtr metadata) {
  return publish(std::move(metadata)).status();
}

template <typename ReaderT>
arrow::Result<typename FormatReaderMetadataCache<ReaderT>::MetadataPtr> FormatReaderMetadataCache<ReaderT>::get_or_open(
    const std::string& key,
    const typename FormatReaderMetadataCache<ReaderT>::MetadataLoader& load_fn,
    const uint64_t estimated_bytes) {
  ARROW_ASSIGN_OR_RAISE(auto cached, get(key));
  if (cached) {
    return cached;
  }
  std::shared_ptr<InFlightLoad> in_flight_load;
  bool owns_in_flight_load = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    // A previous flight may have published since the first lookup.
    auto pin = cache_->Get(key);
    if (!pin || *pin) {
      // Unpin can notify waiters inline, including on a wrapper allocation failure.
      lock.unlock();
      return FromPin(std::move(pin));
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
  auto result = [&]() -> MetadataResult {
    try {
      ARROW_ASSIGN_OR_RAISE(const auto reservation, reserve_for_load(estimated_bytes));
      auto refund = folly::makeGuard([&] { release_load_reservation(reservation); });
      ARROW_ASSIGN_OR_RAISE(auto metadata, load_fn());
      return publish(std::move(metadata), reservation);
    } catch (const std::exception& e) {
      return arrow::Status::UnknownError("Exception while loading format reader metadata: ", e.what());
    } catch (...) {
      return arrow::Status::UnknownError("Unknown exception while loading format reader metadata");
    }
  }();

  // Only the leader completes the flight. Its reservation has already been returned.
  if (owns_in_flight_load) {
    return complete_load(key, in_flight_load, std::move(result));
  }
  return result;
}

template <typename ReaderT>
folly::SemiFuture<typename FormatReaderMetadataCache<ReaderT>::MetadataResult>
FormatReaderMetadataCache<ReaderT>::get_or_open_async(
    const std::string& key,
    const typename FormatReaderMetadataCache<ReaderT>::AsyncMetadataLoader& load_fn,
    const uint64_t estimated_bytes) {
  // Keep the cache alive for the whole deferred operation. Lookup and loading do
  // not start until the returned SemiFuture is consumed.
  auto self = this->shared_from_this();

  return folly::makeSemiFuture().deferValue([self = std::move(self), key, load_fn,
                                             estimated_bytes](folly::Unit) -> folly::SemiFuture<MetadataResult> {
    auto cached = self->get(key);
    if (!cached.ok() || *cached) {
      return folly::makeSemiFuture(std::move(cached));
    }
    std::shared_ptr<InFlightLoad> in_flight_load;
    {
      std::unique_lock<std::mutex> lock(self->mutex_);

      // Recheck the cache before registering a new flight.
      auto pin = self->cache_->Get(key);
      if (!pin || *pin) {
        lock.unlock();
        return folly::makeSemiFuture(FromPin(std::move(pin)));
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
      // Copy the key before reserving so installing cleanup after a grant cannot allocate.
      auto cleanup_key = key;
      const auto admission = self->reserve_for_load(estimated_bytes);
      if (!admission.ok()) {
        return folly::makeSemiFuture(self->complete_load(key, in_flight_load, admission.status()));
      }
      const auto reservation = *admission;
      // Timed SemiFuture waits can discard their deferred continuation. Release
      // its flight and notify followers even when that continuation never runs.
      auto cleanup = folly::makeGuard([self, key = std::move(cleanup_key), in_flight_load, reservation]() noexcept {
        self->release_load_reservation(reservation);
        if (!in_flight_load->continuation_attached.load(std::memory_order_acquire)) {
          return;
        }
        auto cleaned = folly::makeTryWith([&] {
          return self->complete_load(key, in_flight_load,
                                     arrow::Status::Cancelled("Async format reader metadata load was abandoned"));
        });
        if (cleaned.hasException()) {
          const auto* error = cleaned.exception().get_exception();
          RAW_LOG(ERROR, "Failed to abandon format reader metadata load: %s",
                  error ? error->what() : "unknown exception");
        }
      });
      auto load_future = load_fn();
      auto pending = std::move(load_future)
                         .defer([self, key, in_flight_load, reservation, cleanup = std::move(cleanup)](
                                    folly::Try<MetadataResult>&& load_try) mutable -> MetadataResult {
                           MetadataResult result = [&]() -> MetadataResult {
                             if (load_try.hasException()) {
                               auto message = load_try.exception().what();
                               return arrow::Status::UnknownError(
                                   "Exception while asynchronously loading format reader metadata: ",
                                   std::string(message.data(), message.size()));
                             }
                             ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(load_try).value());
                             return self->publish(std::move(metadata), reservation);
                           }();
                           cleanup.dismiss();
                           self->release_load_reservation(reservation);
                           return self->complete_load(key, in_flight_load, std::move(result));
                         });
      // Attachment failures belong to the catch below, not to cancellation.
      in_flight_load->continuation_attached.store(true, std::memory_order_release);
      return pending;
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
  const auto& status = load_result.status();
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // A discarded continuation and a submission failure can both attempt to
    // clean up the same flight. Only its first completion may fulfill waiters.
    if (in_flight_load->done) {
      if (!in_flight_load->status.ok()) {
        return in_flight_load->status;
      }
      return in_flight_load->metadata;
    }

    in_flight_load->status = status;
    if (load_result.ok()) {
      in_flight_load->metadata = *load_result;
    }
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
  in_flight_load->async_result.setValue(load_result);
  return load_result;
}

template class FormatReaderMetadataCache<parquet::ParquetFormatReader>;
template class FormatReaderMetadataCache<vortex::VortexFormatReader>;
template class FormatReaderMetadataCache<lance::LanceTableReader>;
template class FormatReaderMetadataCache<iceberg::IcebergFormatReader>;
template class FormatReaderMetadataCache<paimon::PaimonFormatReader>;

}  // namespace milvus_storage
