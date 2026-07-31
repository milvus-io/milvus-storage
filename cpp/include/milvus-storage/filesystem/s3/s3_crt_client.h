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

#pragma once

#ifdef WITH_CRT

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <arrow/result.h>

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>

#include "milvus-storage/filesystem/s3/s3_client_builder.h"

namespace milvus_storage {

class S3CrtClientHolder;

template <>
struct ClientBuilderTraits<Aws::S3Crt::S3CrtClient> {
  using ConfigType = Aws::S3Crt::S3CrtClientConfiguration;
  using HolderType = S3CrtClientHolder;
};

template <>
arrow::Result<std::shared_ptr<S3CrtClientHolder>> ClientBuilder<Aws::S3Crt::S3CrtClient>::BuildClient(
    std::optional<arrow::io::IOContext> io_context, std::shared_ptr<FilesystemMetrics> metrics);

class S3CrtClientOperationState;
class S3CrtClientFinalizer;

/// A move-only lease for one CRT client operation.
///
/// The lease deliberately contains only a non-owning client pointer and
/// operation state. In particular, it must never own S3CrtClient,
/// S3CrtClientHolder, or ObjectCrtInputFile. This makes it safe to release the
/// lease on a CRT callback thread without running the CRT client destructor
/// there.
class S3CrtClientLease {
  public:
  S3CrtClientLease() = default;
  S3CrtClientLease(const S3CrtClientLease&) = delete;
  S3CrtClientLease& operator=(const S3CrtClientLease&) = delete;
  S3CrtClientLease(S3CrtClientLease&& other) noexcept;
  S3CrtClientLease& operator=(S3CrtClientLease&& other) noexcept;
  ~S3CrtClientLease();

  Aws::S3Crt::S3CrtClient* operator->() const;

  protected:
  friend class S3CrtClientHolder;
  void Release();

  Aws::S3Crt::S3CrtClient* client_ = nullptr;
  std::shared_ptr<S3CrtClientOperationState> operation_state_;
};

class S3CrtClientHolder {
  public:
  /// Acquire a non-owning client pointer protected by an operation lease.
  /// No lock is held while the caller uses the client.
  arrow::Result<S3CrtClientLease> Acquire();
  /// The last holder reference must not be released from a native CRT callback.
  ~S3CrtClientHolder();
  std::shared_ptr<FilesystemMetrics> GetMetrics() const;

  protected:
  friend class S3CrtClientFinalizer;
  S3CrtClientHolder(std::shared_ptr<S3CrtClientFinalizer> finalizer,
                    std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                    std::shared_ptr<FilesystemMetrics> metrics);
  void Finalize();

  std::shared_ptr<S3CrtClientFinalizer> finalizer_;
  std::shared_ptr<S3CrtClientOperationState> operation_state_;
  // The holder is the sole shared owner of the client. Operation leases must
  // never copy this shared_ptr.
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

class S3CrtClientFinalizer : public std::enable_shared_from_this<S3CrtClientFinalizer> {
  using ClientHolderList = std::vector<std::weak_ptr<S3CrtClientHolder>>;

  public:
  arrow::Result<std::shared_ptr<S3CrtClientHolder>> AddClient(std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                                                              std::shared_ptr<FilesystemMetrics> metrics);
  /// Close all holders and wait until every CRT client destructor has returned.
  /// This is an S3 lifecycle operation and must not be called from a CRT
  /// callback.
  void Finalize();

  protected:
  friend class S3CrtClientHolder;
  void ClientDestroyed();

  std::mutex mutex_;
  std::condition_variable cv_;
  ClientHolderList holders_;
  std::size_t live_clients_ = 0;
  std::atomic<bool> finalized_{false};
};

std::shared_ptr<S3CrtClientFinalizer> GetCrtClientFinalizer();

}  // namespace milvus_storage

#endif  // WITH_CRT
