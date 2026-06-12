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

#include "milvus-storage/filesystem/s3/s3_client_builder.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>

#include <arrow/filesystem/path_util.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/thread_pool.h>

#include <aws/core/auth/signer/AWSAuthV4Signer.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/s3/S3Client.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/common/path_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/util_internal.h"

using ::arrow::Result;
using ::arrow::Status;
using ::milvus_storage::fs::internal::ConnectRetryStrategy;
using ::milvus_storage::fs::internal::FromAwsString;
using ::milvus_storage::fs::internal::ToAwsString;

namespace milvus_storage {

namespace fs::internal {

class WrappedRetryStrategy : public Aws::Client::RetryStrategy {
  public:
  explicit WrappedRetryStrategy(const std::shared_ptr<S3RetryStrategy>& s3_retry_strategy)
      : s3_retry_strategy_(s3_retry_strategy) {}

  [[nodiscard]] bool ShouldRetry(const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
                                 long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return s3_retry_strategy_->ShouldRetry(detail, static_cast<int64_t>(attempted_retries));
  }

  long CalculateDelayBeforeNextRetry(  // NOLINT runtime/int
      const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
      long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return static_cast<long>(  // NOLINT runtime/int
        s3_retry_strategy_->CalculateDelayBeforeNextRetry(detail, static_cast<int64_t>(attempted_retries)));
  }

  private:
  template <typename ErrorType>
  static S3RetryStrategy::AWSErrorDetail ErrorToDetail(const Aws::Client::AWSError<ErrorType>& error) {
    S3RetryStrategy::AWSErrorDetail detail;
    detail.error_type = static_cast<int>(error.GetErrorType());
    detail.message = std::string(FromAwsString(error.GetMessage()));
    detail.exception_name = std::string(FromAwsString(error.GetExceptionName()));
    detail.should_retry = error.ShouldRetry();
    return detail;
  }

  std::shared_ptr<S3RetryStrategy> s3_retry_strategy_;
};

std::shared_ptr<Aws::Client::RetryStrategy> MakeWrappedRetryStrategy(
    const std::shared_ptr<S3RetryStrategy>& s3_retry_strategy) {
  return std::make_shared<WrappedRetryStrategy>(s3_retry_strategy);
}

}  // namespace fs::internal

ClientBuilderBase::ClientBuilderBase(S3Options options) : options_(std::move(options)) {}

const S3Options& ClientBuilderBase::options() const { return options_; }

const std::shared_ptr<Aws::Auth::AWSCredentialsProvider>& ClientBuilderBase::credentials_provider() const {
  return credentials_provider_;
}

arrow::Status ClientBuilderBase::PrepareClientConfig(Aws::Client::ClientConfiguration* client_config,
                                                     std::optional<arrow::io::IOContext> io_context) {
  credentials_provider_ = options_.credentials_provider;

  // HOTFIX: Prevent nullptr crash in GCP IAM and other race condition scenarios
  // Root cause: Race condition between S3Options construction and ConfigureAccessKey call
  // TODO: Investigate and fix the root cause of the race condition
  if (!credentials_provider_) {
    LOG_STORAGE_ERROR_ << "[HOTFIX] credentials_provider is nullptr! "
                       << "This indicates a race condition or missing initialization. "
                       << "Using AnonymousCredentialsProvider as fallback. "
                       << "Please report this error with stack trace.";
    return arrow::Status::Invalid("credentials_provider is nullptr");
  }

  if (!options_.region.empty()) {
    client_config->region = ToAwsString(options_.region);
  }
  if (options_.request_timeout > 0) {
    // Use ceil() to avoid setting it to 0 as that probably means no timeout.
    client_config->requestTimeoutMs = static_cast<long>(ceil(options_.request_timeout * 1000));  // NOLINT runtime/int
  }
  if (options_.connect_timeout > 0) {
    client_config->connectTimeoutMs = static_cast<long>(ceil(options_.connect_timeout * 1000));  // NOLINT runtime/int
  }

  client_config->endpointOverride = ToAwsString(options_.endpoint_override);
  if (options_.scheme == "http") {
    client_config->scheme = Aws::Http::Scheme::HTTP;
    client_config->verifySSL = false;
  } else if (options_.scheme == "https") {
    client_config->scheme = Aws::Http::Scheme::HTTPS;
    client_config->verifySSL = true;
  } else {
    return arrow::Status::Invalid("Invalid S3 connection scheme '", options_.scheme, "'");
  }
  if (!arrow::fs::internal::global_options.tls_ca_file_path.empty()) {
    client_config->caFile = ToAwsString(arrow::fs::internal::global_options.tls_ca_file_path);
    client_config->verifySSL = false;
  }
  if (!arrow::fs::internal::global_options.tls_ca_dir_path.empty()) {
    client_config->caPath = ToAwsString(arrow::fs::internal::global_options.tls_ca_dir_path);
  }

  // Set proxy options if provided
  if (!options_.proxy_options.scheme.empty()) {
    if (options_.proxy_options.scheme == "http") {
      client_config->proxyScheme = Aws::Http::Scheme::HTTP;
    } else if (options_.proxy_options.scheme == "https") {
      client_config->proxyScheme = Aws::Http::Scheme::HTTPS;
    } else {
      return arrow::Status::Invalid("Invalid proxy connection scheme '", options_.proxy_options.scheme, "'");
    }
  }
  if (!options_.proxy_options.host.empty()) {
    client_config->proxyHost = ToAwsString(options_.proxy_options.host);
  }
  if (options_.proxy_options.port != -1) {
    client_config->proxyPort = options_.proxy_options.port;
  }
  if (!options_.proxy_options.username.empty()) {
    client_config->proxyUserName = ToAwsString(options_.proxy_options.username);
  }
  if (!options_.proxy_options.password.empty()) {
    client_config->proxyPassword = ToAwsString(options_.proxy_options.password);
  }

  if (io_context) {
    // TODO: Once ARROW-15035 is done we can get rid of the "at least 25" fallback
    client_config->maxConnections = std::max(io_context->executor()->GetCapacity(), 25);
  }

  client_config->maxConnections = std::max(client_config->maxConnections, options_.max_connections);

  // Non-AWS S3-compatible APIs (GCP, Aliyun OSS, Tencent COS, Huawei OBS) do
  // not accept the extra x-amz-checksum-* headers / aws-chunked streaming that
  // AWS SDK >= 1.11.x sends by default (WHEN_SUPPORTED). Restrict to
  // WHEN_REQUIRED so the SDK only adds checksums when the API mandates them.
  if (options_.cloud_provider == kCloudProviderGCP || options_.cloud_provider == kCloudProviderAliyun ||
      options_.cloud_provider == kCloudProviderTencent || options_.cloud_provider == kCloudProviderHuawei) {
    client_config->checksumConfig.requestChecksumCalculation = Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED;
    client_config->checksumConfig.responseChecksumValidation = Aws::Client::ResponseChecksumValidation::WHEN_REQUIRED;
  }

  return Status::OK();
}

template <>
arrow::Result<std::shared_ptr<S3ClientHolder>> ClientBuilder<S3Client>::BuildClient(
    std::optional<arrow::io::IOContext> io_context, std::shared_ptr<FilesystemMetrics> metrics) {
  (void)metrics;
  ARROW_RETURN_NOT_OK(PrepareClientConfig(io_context));

  if (options_.retry_strategy) {
    client_config_.retryStrategy = fs::internal::MakeWrappedRetryStrategy(options_.retry_strategy);
  } else {
    client_config_.retryStrategy = std::make_shared<ConnectRetryStrategy>();
  }

  const bool use_virtual_addressing = options_.endpoint_override.empty() || options_.force_virtual_addressing;

#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
  client_config_.useVirtualAddressing = use_virtual_addressing;
  auto endpoint_provider = EndpointProviderCache::Instance()->Lookup(client_config_);
  auto client = std::make_shared<S3Client>(credentials_provider_, endpoint_provider, client_config_);
#else
  auto client =
      std::make_shared<S3Client>(credentials_provider_, client_config_,
                                 Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, use_virtual_addressing);
#endif
  client->s3_retry_strategy_ = options_.retry_strategy;
  return GetClientFinalizer()->AddClient(std::move(client));
}
}  // namespace milvus_storage
