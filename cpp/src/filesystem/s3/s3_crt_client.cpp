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
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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

// ------------ Implementation of S3CrtClientHolder ------------
Aws::S3Crt::S3CrtClient* S3CrtClientLock::get() { return client_.get(); }
Aws::S3Crt::S3CrtClient* S3CrtClientLock::operator->() { return client_.get(); }
S3CrtClientLock S3CrtClientLock::Move() { return std::move(*this); }

S3CrtClientHolder::S3CrtClientHolder(std::weak_ptr<S3CrtClientFinalizer> finalizer,
                                     std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                                     std::shared_ptr<FilesystemMetrics> metrics)
    : finalizer_(std::move(finalizer)), client_(std::move(client)), metrics_(std::move(metrics)) {}

arrow::Result<S3CrtClientLock> S3CrtClientHolder::Lock() {
  std::shared_ptr<S3CrtClientFinalizer> finalizer;
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client;
  {
    std::unique_lock lock(mutex_);
    finalizer = finalizer_.lock();
    client = client_;
  }
  if (!finalizer) {
    return ErrorS3Finalized();
  }

  S3CrtClientLock client_lock;
  client_lock.lock_ = finalizer->LockShared();
  if (finalizer->finalized_) {
    return ErrorS3Finalized();
  }
  DCHECK(client) << "inconsistent S3CrtClientHolder";
  client_lock.client_ = std::move(client);
  return client_lock;
}

void S3CrtClientHolder::Finalize() {
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client;
  {
    std::unique_lock lock(mutex_);
    client = std::move(client_);
  }
}

std::shared_ptr<FilesystemMetrics> S3CrtClientHolder::GetMetrics() const { return metrics_; }

using CrtClientHolderList = std::vector<std::weak_ptr<S3CrtClientHolder>>;

arrow::Result<std::shared_ptr<S3CrtClientHolder>> S3CrtClientFinalizer::AddClient(
    std::shared_ptr<Aws::S3Crt::S3CrtClient> client, std::shared_ptr<FilesystemMetrics> metrics) {
  std::unique_lock lock(mutex_);
  if (finalized_) {
    return ErrorS3Finalized();
  }

  auto holder = std::make_shared<S3CrtClientHolder>(shared_from_this(), std::move(client), std::move(metrics));

  auto end = std::remove_if(holders_.begin(), holders_.end(),
                            [](const std::weak_ptr<S3CrtClientHolder>& holder) { return holder.expired(); });
  holders_.erase(end, holders_.end());
  holders_.emplace_back(holder);
  return holder;
}

void S3CrtClientFinalizer::Finalize() {
  std::unique_lock lock(mutex_);
  finalized_ = true;

  CrtClientHolderList finalizing = std::move(holders_);
  lock.unlock();

  for (auto&& weak_holder : finalizing) {
    auto holder = weak_holder.lock();
    if (holder) {
      holder->Finalize();
    }
  }
}

std::shared_lock<std::shared_mutex> S3CrtClientFinalizer::LockShared() { return std::shared_lock(mutex_); }

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
