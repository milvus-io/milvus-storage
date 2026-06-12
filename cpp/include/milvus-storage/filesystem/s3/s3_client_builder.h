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

#include <memory>
#include <optional>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/io/interfaces.h>

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/client/ClientConfiguration.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

namespace milvus_storage {

template <typename ClientT>
struct ClientBuilderTraits;

namespace client_builder_internal {

inline Aws::Client::ClientConfigurationInitValues MakeClientConfigurationInitValues(const S3Options& options) {
  if (options.cloud_provider == kCloudProviderGCP || options.cloud_provider == kCloudProviderAliyun ||
      options.cloud_provider == kCloudProviderTencent || options.cloud_provider == kCloudProviderHuawei) {
    return Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true};
  }
  return Aws::Client::ClientConfigurationInitValues();
}

}  // namespace client_builder_internal

class ClientBuilderBase {
  public:
  explicit ClientBuilderBase(S3Options options);

  const S3Options& options() const;

  const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& credentials_provider() const;

  protected:
  arrow::Status PrepareClientConfig(Aws::Client::ClientConfiguration* client_config,
                                    std::optional<arrow::io::IOContext> io_context);

  S3Options options_;
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider_;
};

template <typename ClientT>
class ClientBuilder : public ClientBuilderBase {
  public:
  using ConfigType = typename ClientBuilderTraits<ClientT>::ConfigType;
  using HolderType = typename ClientBuilderTraits<ClientT>::HolderType;

  explicit ClientBuilder(S3Options options)
      : ClientBuilderBase(std::move(options)),
        client_config_(client_builder_internal::MakeClientConfigurationInitValues(options_)) {}

  const ConfigType& config() const { return client_config_; }

  ConfigType* mutable_config() { return &client_config_; }

  arrow::Status PrepareClientConfig(std::optional<arrow::io::IOContext> io_context) {
    return ClientBuilderBase::PrepareClientConfig(&client_config_, io_context);
  }

  arrow::Result<std::shared_ptr<HolderType>> BuildClient(std::optional<arrow::io::IOContext> io_context = std::nullopt,
                                                         std::shared_ptr<FilesystemMetrics> metrics = nullptr);

  private:
  ConfigType client_config_;
};

ClientBuilder(S3Options) -> ClientBuilder<S3Client>;

template <>
struct ClientBuilderTraits<S3Client> {
#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
  using ConfigType = Aws::S3::S3ClientConfiguration;
#else
  using ConfigType = Aws::Client::ClientConfiguration;
#endif
  using HolderType = S3ClientHolder;
};

template <>
arrow::Result<std::shared_ptr<S3ClientHolder>> ClientBuilder<S3Client>::BuildClient(
    std::optional<arrow::io::IOContext> io_context, std::shared_ptr<FilesystemMetrics> metrics);

}  // namespace milvus_storage
