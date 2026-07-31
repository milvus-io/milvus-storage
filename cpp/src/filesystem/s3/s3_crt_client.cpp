// Copyright 2024 Zilliz
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

#include "milvus-storage/filesystem/s3/s3_crt_client.h"

#ifdef WITH_CRT

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <arrow/status.h>
#include <arrow/util/logging.h>

#include <aws/core/Aws.h>
#include <aws/core/auth/signer/AWSAuthV4Signer.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>

#include "milvus-storage/filesystem/s3/s3_internal.h"

namespace milvus_storage {

namespace fs::internal {

// WrappedRetryStrategy is implemented in s3_client_builder.cpp with the client
// construction glue. CRT only needs this factory to preserve
// S3Options::retry_strategy without exposing a private adapter through another
// header.
std::shared_ptr<Aws::Client::RetryStrategy> MakeWrappedRetryStrategy(
    const std::shared_ptr<S3RetryStrategy>& s3_retry_strategy);

}  // namespace fs::internal

namespace {

inline arrow::Status ErrorS3Finalized() { return arrow::Status::Invalid("S3 subsystem is finalized"); }

}  // namespace

// Per-holder operation gate shared with every outstanding lease.
//
// Separating this block from S3CrtClientHolder is the core ownership rule:
//
//   holder --shared--> operation state <--shared-- lease
//      |                                         |
//      | sole shared owner                       | raw borrow
//      v                                         v
//   S3CrtClient <--------------------------------+
//
// A lease may outlive the point at which holder destruction starts, so it must
// keep the synchronization block alive. It must not keep the holder or client
// alive: releasing the last such owner on a native CRT callback thread could
// run S3CrtClient::~S3CrtClient there, and that destructor waits for CRT
// shutdown callbacks.
//
// Locking rules:
//   * mutex protects both fields below.
//   * no S3 operation executes while mutex is held.
//   * the CRT client destructor executes only after mutex is released.
class S3CrtClientOperationState {
  public:
  std::mutex mutex;
  std::condition_variable cv;
  // Number of leases that may currently dereference their raw client pointer.
  std::size_t active_operations = 0;
  // One-way admission gate: false -> true when holder finalization starts.
  bool closing = false;
};

// ------------ Implementation of S3CrtClientConstructionLease ------------
// RAII reservation covering the entire client factory call, including factory
// cleanup on success and every early-return/error path.
//
//   AddClient                       Finalize
//   ---------                       --------
//   constructing_clients_++         finalized_ = true
//   create guard                    wait constructing_clients_ == 0
//   unlock finalizer mutex                    ^
//   run factory                               |
//   destroy factory captures                  |
//   register or return error         guard/manual decrement + notify
//
// The factory is deliberately invoked without the finalizer mutex held. The
// reservation, rather than that mutex, prevents global shutdown from passing
// client construction.
class S3CrtClientConstructionLease {
  public:
  S3CrtClientConstructionLease(const S3CrtClientConstructionLease&) = delete;
  S3CrtClientConstructionLease& operator=(const S3CrtClientConstructionLease&) = delete;
  S3CrtClientConstructionLease(S3CrtClientConstructionLease&&) = delete;
  S3CrtClientConstructionLease& operator=(S3CrtClientConstructionLease&&) = delete;
  ~S3CrtClientConstructionLease();

  private:
  friend class S3CrtClientFinalizer;
  explicit S3CrtClientConstructionLease(std::shared_ptr<S3CrtClientFinalizer> finalizer)
      : finalizer_(std::move(finalizer)) {}

  std::shared_ptr<S3CrtClientFinalizer> finalizer_;
};

// On a failed construction path finalizer_ is still set, so this destructor
// releases the reservation. Successful registration disarms the guard and
// decrements the same counter atomically with publishing the new live client.
S3CrtClientConstructionLease::~S3CrtClientConstructionLease() {
  auto finalizer = std::move(finalizer_);
  if (!finalizer) {
    return;
  }

  bool notify = false;
  {
    std::lock_guard lock(finalizer->mutex_);
    DCHECK_GT(finalizer->constructing_clients_, 0);
    notify = --finalizer->constructing_clients_ == 0;
  }
  if (notify) {
    finalizer->cv_.notify_all();
  }
}

// ------------ Implementation of S3CrtClientHolder ------------
S3CrtClientLease::S3CrtClientLease(S3CrtClientLease&& other) noexcept
    : client_(std::exchange(other.client_, nullptr)), operation_state_(std::move(other.operation_state_)) {}

S3CrtClientLease& S3CrtClientLease::operator=(S3CrtClientLease&& other) noexcept {
  if (this != &other) {
    Release();
    client_ = std::exchange(other.client_, nullptr);
    operation_state_ = std::move(other.operation_state_);
  }
  return *this;
}

S3CrtClientLease::~S3CrtClientLease() { Release(); }

Aws::S3Crt::S3CrtClient* S3CrtClientLease::operator->() const { return client_; }

void S3CrtClientLease::Release() {
  // Clear the borrowed pointer before publishing that this operation is done.
  // Moving operation_state_ out also makes repeated Release() calls harmless,
  // which is required by move assignment and moved-from destruction.
  client_ = nullptr;
  auto operation_state = std::move(operation_state_);
  if (!operation_state) {
    return;
  }

  bool notify = false;
  {
    std::lock_guard lock(operation_state->mutex);
    DCHECK_GT(operation_state->active_operations, 0);
    notify = --operation_state->active_operations == 0;
  }
  if (notify) {
    operation_state->cv.notify_all();
  }
}

S3CrtClientHolder::S3CrtClientHolder(std::shared_ptr<S3CrtClientFinalizer> finalizer,
                                     std::shared_ptr<FilesystemMetrics> metrics)
    : finalizer_(std::move(finalizer)),
      operation_state_(std::make_shared<S3CrtClientOperationState>()),
      metrics_(std::move(metrics)) {}

S3CrtClientHolder::~S3CrtClientHolder() { Finalize(); }

// Holder state transition guarded by operation_state_->mutex:
//
//   OPEN, active=N -- Acquire() --> OPEN, active=N+1
//   OPEN           -- Finalize() --> CLOSING -- active=0 --> client destroy
//   CLOSING        -- Acquire() --> rejected
//
// finalized_ is also checked because global shutdown closes admission before
// it has collected and finalized every individual holder.
arrow::Result<S3CrtClientLease> S3CrtClientHolder::Acquire() {
  S3CrtClientLease lease;
  {
    std::lock_guard operation_lock(operation_state_->mutex);
    if (operation_state_->closing || finalizer_->finalized_.load(std::memory_order_acquire)) {
      return ErrorS3Finalized();
    }
    DCHECK(client_) << "inconsistent S3CrtClientHolder";
    ++operation_state_->active_operations;
    lease.client_ = client_.get();
    lease.operation_state_ = operation_state_;
  }
  return lease;
}

// Finalization deliberately separates waiting, ownership transfer, and native
// destruction:
//
//   finalizing/destructor thread             operation/CRT callback threads
//   ----------------------------             ------------------------------
//   closing = true
//   wait(active_operations == 0)   <-------  lease Release(): active--, notify
//   move client_ out
//   unlock operation mutex
//   client.reset()
//   ClientDestroyed()
//
// No new lease can start after closing is set. Existing leases keep only the
// operation state alive and eventually wake this waiter. client.reset() is
// outside the operation mutex because the SDK destructor may block while CRT
// completes its own shutdown callbacks.
void S3CrtClientHolder::Finalize() {
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client;
  {
    std::unique_lock lock(operation_state_->mutex);
    operation_state_->closing = true;
    operation_state_->cv.wait(lock, [this] { return operation_state_->active_operations == 0; });
    client = std::move(client_);
  }
  if (!client) {
    return;
  }

  // S3CrtClient::~S3CrtClient waits for the native CRT client shutdown
  // callback. This runs only after every operation lease has left its callback
  // and always on the thread finalizing/destroying the holder.
  client.reset();
  finalizer_->ClientDestroyed();
}

std::shared_ptr<FilesystemMetrics> S3CrtClientHolder::GetMetrics() const { return metrics_; }

// Construction is published as one transaction under mutex_:
//
//   before factory:  constructing_clients_++
//   after success:   holders_ += holder
//                    holder->client_ = client
//                    live_clients_++
//                    constructing_clients_--
//
// Finalize() cannot observe a client between construction and registration. On
// failure the construction guard performs only the final decrement, so a null
// or multiply-owned client is never counted as live.
arrow::Result<std::shared_ptr<S3CrtClientHolder>> S3CrtClientFinalizer::AddClient(
    ClientFactory make_client, std::shared_ptr<FilesystemMetrics> metrics) {
  auto finalizer = shared_from_this();
  {
    std::lock_guard lock(mutex_);
    if (finalized_.load(std::memory_order_acquire)) {
      return ErrorS3Finalized();
    }
    ++constructing_clients_;
  }
  S3CrtClientConstructionLease construction(finalizer);

  // Destroy the std::function and all of its captures outside mutex_. Captures
  // may perform arbitrary cleanup; that cleanup remains inside the construction
  // reservation but never blocks other finalizer state transitions.
  ClientFactory factory = std::move(make_client);
  make_client = nullptr;
  ARROW_ASSIGN_OR_RAISE(auto client, factory());
  factory = nullptr;
  if (!client) {
    return arrow::Status::Invalid("S3 CRT client factory returned a null client");
  }
  if (client.use_count() != 1) {
    return arrow::Status::Invalid("S3CrtClientHolder must be the sole shared owner");
  }
  auto holder = std::shared_ptr<S3CrtClientHolder>(new S3CrtClientHolder(finalizer, std::move(metrics)));

  bool notify = false;
  {
    std::lock_guard lock(mutex_);
    auto end = std::remove_if(holders_.begin(), holders_.end(),
                              [](const std::weak_ptr<S3CrtClientHolder>& item) { return item.expired(); });
    holders_.erase(end, holders_.end());
    holders_.emplace_back(holder);

    holder->client_ = std::move(client);
    ++live_clients_;
    construction.finalizer_.reset();
    notify = --constructing_clients_ == 0;
  }
  if (notify) {
    cv_.notify_all();
  }
  return holder;
}

// Global finalization has three phases and never holds mutex_ while waiting on
// a holder or destroying a native client:
//
//   Phase 1: close admission and stabilize the registry
//     finalized_ = true
//     wait constructing_clients_ == 0
//     weak holder registry --lock()--> local strong holder list
//
//   Phase 2: drain each holder without mutex_
//     closing = true -> wait leases -> destroy client -> ClientDestroyed()
//
//   Phase 3: wait live_clients_ == 0
//
// The final counter wait is required even after every lockable weak holder has
// been drained. A weak entry can already be expired while its holder destructor
// is blocked on an operation lease or inside the native client destructor; that
// client remains live until ClientDestroyed() runs.
//
// Concurrent callers share the same drain. Only the caller that changes
// finalized_ from false to true consumes holders_; all callers still wait for
// both counters to reach zero.
void S3CrtClientFinalizer::Finalize() {
  std::vector<std::shared_ptr<S3CrtClientHolder>> finalizing;
  {
    std::unique_lock lock(mutex_);
    const bool first_finalizer = !finalized_.exchange(true, std::memory_order_acq_rel);
    cv_.wait(lock, [this] { return constructing_clients_ == 0; });
    if (first_finalizer) {
      finalizing.reserve(holders_.size());
      for (auto&& weak_holder : holders_) {
        if (auto holder = weak_holder.lock()) {
          finalizing.emplace_back(std::move(holder));
        }
      }
      holders_.clear();
    }
  }

  for (auto&& holder : finalizing) {
    holder->Finalize();
  }

  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return live_clients_ == 0; });
}

void S3CrtClientFinalizer::ClientDestroyed() {
  // This callback runs after S3CrtClient::~S3CrtClient has returned, so zero
  // means every registered native client is fully gone and Aws::ShutdownAPI()
  // may proceed.
  bool notify = false;
  {
    std::lock_guard lock(mutex_);
    DCHECK_GT(live_clients_, 0);
    notify = --live_clients_ == 0;
  }
  if (notify) {
    cv_.notify_all();
  }
}

std::shared_ptr<S3CrtClientFinalizer> GetCrtClientFinalizer() {
  static auto finalizer = std::make_shared<S3CrtClientFinalizer>();
  return finalizer;
}

template <>
arrow::Result<std::shared_ptr<S3CrtClientHolder>> ClientBuilder<Aws::S3Crt::S3CrtClient>::BuildClient(
    std::optional<arrow::io::IOContext> io_context, std::shared_ptr<FilesystemMetrics> metrics) {
  if (!metrics) {
    metrics = std::make_shared<FilesystemMetrics>();
  }
  // Configuration and native client construction stay inside AddClient's
  // reserved factory window. Global finalization therefore cannot reach
  // Aws::ShutdownAPI() while either step is still using AWS runtime state.
  return GetCrtClientFinalizer()->AddClient(
      [this, io_context = std::move(io_context)]() mutable -> arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>> {
        ARROW_RETURN_NOT_OK(PrepareClientConfig(std::move(io_context)));

        if (options_.retry_strategy) {
          client_config_.retryStrategy = fs::internal::MakeWrappedRetryStrategy(options_.retry_strategy);
        } else {
          client_config_.retryStrategy = std::make_shared<fs::internal::ConnectRetryStrategy>();
        }

        const bool use_virtual_addressing = options_.endpoint_override.empty() || options_.force_virtual_addressing;
        client_config_.useVirtualAddressing = use_virtual_addressing;
        // Raise the CRT target from its SDK default so small concurrent range reads
        // get enough connection budget for the Vortex reader workload.
        // TODO: make this configurable instead of using a fixed workload-specific default.
        client_config_.throughputTargetGbps = 50.0;

        return std::make_shared<Aws::S3Crt::S3CrtClient>(credentials_provider_, client_config_,
                                                         client_config_.payloadSigningPolicy,
                                                         client_config_.useVirtualAddressing);
      },
      std::move(metrics));
}

}  // namespace milvus_storage

#endif  // WITH_CRT
