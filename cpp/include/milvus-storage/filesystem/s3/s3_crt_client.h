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

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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

class S3CrtClientLock {
  public:
  Aws::S3Crt::S3CrtClient* get();
  Aws::S3Crt::S3CrtClient* operator->();
  S3CrtClientLock Move();

  protected:
  friend class S3CrtClientHolder;
  std::shared_lock<std::shared_mutex> lock_;
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client_;
};

class S3CrtClientFinalizer;

class S3CrtClientHolder {
  public:
  arrow::Result<S3CrtClientLock> Lock();
  S3CrtClientHolder(std::weak_ptr<S3CrtClientFinalizer> finalizer,
                    std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                    std::shared_ptr<FilesystemMetrics> metrics);
  void Finalize();
  std::shared_ptr<FilesystemMetrics> GetMetrics() const;

  protected:
  std::mutex mutex_;
  std::weak_ptr<S3CrtClientFinalizer> finalizer_;
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

class S3CrtClientFinalizer : public std::enable_shared_from_this<S3CrtClientFinalizer> {
  using ClientHolderList = std::vector<std::weak_ptr<S3CrtClientHolder>>;

  public:
  arrow::Result<std::shared_ptr<S3CrtClientHolder>> AddClient(std::shared_ptr<Aws::S3Crt::S3CrtClient> client,
                                                              std::shared_ptr<FilesystemMetrics> metrics);
  void Finalize();
  std::shared_lock<std::shared_mutex> LockShared();

  protected:
  friend class S3CrtClientHolder;
  std::shared_mutex mutex_;
  ClientHolderList holders_;
  bool finalized_ = false;
};

std::shared_ptr<S3CrtClientFinalizer> GetCrtClientFinalizer();

}  // namespace milvus_storage

#endif  // WITH_CRT
