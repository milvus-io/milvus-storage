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

class S3CrtClientOperationState {
  public:
  std::mutex mutex;
  std::condition_variable cv;
  std::size_t active_operations = 0;
  bool closing = false;
};

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
                                     std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                                     std::shared_ptr<FilesystemMetrics> metrics)
    : finalizer_(std::move(finalizer)),
      operation_state_(std::make_shared<S3CrtClientOperationState>()),
      client_(std::move(client)),
      metrics_(std::move(metrics)) {}

S3CrtClientHolder::~S3CrtClientHolder() { Finalize(); }

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

arrow::Result<std::shared_ptr<S3CrtClientHolder>> S3CrtClientFinalizer::AddClient(
    std::shared_ptr<Aws::S3Crt::S3CrtClient> client, std::shared_ptr<FilesystemMetrics> metrics) {
  std::lock_guard lock(mutex_);
  if (finalized_.load(std::memory_order_acquire)) {
    return ErrorS3Finalized();
  }
  DCHECK(client);
  DCHECK_EQ(client.use_count(), 1) << "S3CrtClientHolder must be the sole shared owner";

  auto holder = std::shared_ptr<S3CrtClientHolder>(
      new S3CrtClientHolder(shared_from_this(), std::move(client), std::move(metrics)));
  ++live_clients_;

  auto end = std::remove_if(holders_.begin(), holders_.end(),
                            [](const std::weak_ptr<S3CrtClientHolder>& holder) { return holder.expired(); });
  holders_.erase(end, holders_.end());
  holders_.emplace_back(holder);
  return holder;
}

void S3CrtClientFinalizer::Finalize() {
  std::vector<std::shared_ptr<S3CrtClientHolder>> finalizing;
  {
    std::lock_guard lock(mutex_);
    if (!finalized_.exchange(true, std::memory_order_acq_rel)) {
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
  ARROW_RETURN_NOT_OK(PrepareClientConfig(io_context));

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

  if (!metrics) {
    metrics = std::make_shared<FilesystemMetrics>();
  }
  auto client = std::make_shared<Aws::S3Crt::S3CrtClient>(
      credentials_provider_, client_config_, client_config_.payloadSigningPolicy, client_config_.useVirtualAddressing);
  return GetCrtClientFinalizer()->AddClient(std::move(client), std::move(metrics));
}

}  // namespace milvus_storage

#endif  // WITH_CRT
