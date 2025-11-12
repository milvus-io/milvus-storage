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

#include "milvus-storage/filesystem/s3/s3_options.h"

#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>

#include <arrow/util/logging.h>
#include <arrow/util/uri.h>
#include <arrow/filesystem/path_util.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

using ::milvus_storage::fs::internal::FromAwsString;
using ::milvus_storage::fs::internal::ToAwsString;

namespace milvus_storage {

static constexpr const char kAwsEndpointUrlEnvVar[] = "AWS_ENDPOINT_URL";
static constexpr const char kAwsEndpointUrlS3EnvVar[] = "AWS_ENDPOINT_URL_S3";
static constexpr const char kSep = '/';

// -----------------------------------------------------------------------
// AwsRetryStrategy implementation

class AwsRetryStrategy : public S3RetryStrategy {
  public:
  explicit AwsRetryStrategy(std::shared_ptr<Aws::Client::RetryStrategy> retry_strategy)
      : retry_strategy_(std::move(retry_strategy)) {}

  bool ShouldRetry(const AWSErrorDetail& detail, int64_t attempted_retries) override {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error = DetailToError(detail);
    return retry_strategy_->ShouldRetry(error, static_cast<long>(attempted_retries));  // NOLINT: runtime/int
  }

  int64_t CalculateDelayBeforeNextRetry(const AWSErrorDetail& detail, int64_t attempted_retries) override {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error = DetailToError(detail);
    return retry_strategy_->CalculateDelayBeforeNextRetry(error,
                                                          static_cast<long>(attempted_retries));  // NOLINT: runtime/int
  }

  private:
  std::shared_ptr<Aws::Client::RetryStrategy> retry_strategy_;
  static Aws::Client::AWSError<Aws::Client::CoreErrors> DetailToError(const S3RetryStrategy::AWSErrorDetail& detail) {
    auto exception_name = ToAwsString(detail.exception_name);
    auto message = ToAwsString(detail.message);
    auto errors = Aws::Client::AWSError<Aws::Client::CoreErrors>(
        static_cast<Aws::Client::CoreErrors>(detail.error_type), exception_name, message, detail.should_retry);
    return errors;
  }
};

std::shared_ptr<S3RetryStrategy> S3RetryStrategy::GetAwsDefaultRetryStrategy(int64_t max_attempts) {
  return std::make_shared<AwsRetryStrategy>(
      std::make_shared<Aws::Client::DefaultRetryStrategy>(static_cast<long>(max_attempts)));  // NOLINT: runtime/int
}

std::shared_ptr<S3RetryStrategy> S3RetryStrategy::GetAwsStandardRetryStrategy(int64_t max_attempts) {
  return std::make_shared<AwsRetryStrategy>(
      std::make_shared<Aws::Client::StandardRetryStrategy>(static_cast<long>(max_attempts)));  // NOLINT: runtime/int
}

// -----------------------------------------------------------------------
// S3Options implementation

S3Options::S3Options() { DCHECK(IsS3Initialized()) << "Must initialize S3 before using S3Options"; }

void S3Options::ConfigureDefaultCredentials() {
  credentials_provider = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  credentials_kind = S3CredentialsKind::Default;
}

void S3Options::ConfigureAnonymousCredentials() {
  credentials_provider = std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>();
  credentials_kind = S3CredentialsKind::Anonymous;
}

void S3Options::ConfigureAccessKey(const std::string& access_key,
                                   const std::string& secret_key,
                                   const std::string& session_token) {
  credentials_provider = std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
      fs::internal::ToAwsString(access_key), fs::internal::ToAwsString(secret_key),
      fs::internal::ToAwsString(session_token));
  credentials_kind = S3CredentialsKind::Explicit;
}

void S3Options::ConfigureAssumeRoleCredentials(const std::string& role_arn,
                                               const std::string& session_name,
                                               const std::string& external_id,
                                               int load_frequency,
                                               const std::shared_ptr<Aws::STS::STSClient>& stsClient) {
  credentials_provider = std::make_shared<Aws::Auth::STSAssumeRoleCredentialsProvider>(
      fs::internal::ToAwsString(role_arn), fs::internal::ToAwsString(session_name),
      fs::internal::ToAwsString(external_id), load_frequency, stsClient);
  credentials_kind = S3CredentialsKind::Role;
}

void S3Options::ConfigureAssumeRoleWithWebIdentityCredentials() {
  // The AWS SDK uses environment variables AWS_DEFAULT_REGION,
  // AWS_ROLE_ARN, AWS_WEB_IDENTITY_TOKEN_FILE and AWS_ROLE_SESSION_NAME
  // to configure the required credentials
  credentials_provider = std::make_shared<Aws::Auth::STSAssumeRoleWebIdentityCredentialsProvider>();
  credentials_kind = S3CredentialsKind::WebIdentity;
}

std::string S3Options::GetAccessKey() const {
  auto credentials = credentials_provider->GetAWSCredentials();
  return std::string(fs::internal::FromAwsString(credentials.GetAWSAccessKeyId()));
}

std::string S3Options::GetSecretKey() const {
  auto credentials = credentials_provider->GetAWSCredentials();
  return std::string(fs::internal::FromAwsString(credentials.GetAWSSecretKey()));
}

std::string S3Options::GetSessionToken() const {
  auto credentials = credentials_provider->GetAWSCredentials();
  return std::string(fs::internal::FromAwsString(credentials.GetSessionToken()));
}

S3Options S3Options::Defaults() {
  S3Options options;
  options.ConfigureDefaultCredentials();
  return options;
}

S3Options S3Options::Anonymous() {
  S3Options options;
  options.ConfigureAnonymousCredentials();
  return options;
}

S3Options S3Options::FromAccessKey(const std::string& access_key,
                                   const std::string& secret_key,
                                   const std::string& session_token) {
  S3Options options;
  options.ConfigureAccessKey(access_key, secret_key, session_token);
  return options;
}

S3Options S3Options::FromAssumeRole(const std::string& role_arn,
                                    const std::string& session_name,
                                    const std::string& external_id,
                                    int load_frequency,
                                    const std::shared_ptr<Aws::STS::STSClient>& stsClient) {
  S3Options options;
  options.role_arn = role_arn;
  options.session_name = session_name;
  options.external_id = external_id;
  options.load_frequency = load_frequency;
  options.ConfigureAssumeRoleCredentials(role_arn, session_name, external_id, load_frequency, stsClient);
  return options;
}

S3Options S3Options::FromAssumeRoleWithWebIdentity() {
  S3Options options;
  options.ConfigureAssumeRoleWithWebIdentityCredentials();
  return options;
}

arrow::Result<S3Options> S3Options::FromUri(const arrow::util::Uri& uri, std::string* out_path) {
  S3Options options;

  const auto bucket = uri.host();
  auto path = uri.path();
  if (bucket.empty()) {
    if (!path.empty()) {
      return arrow::Status::Invalid("Missing bucket name in S3 URI");
    }
  } else {
    if (path.empty()) {
      path = bucket;
    } else {
      if (path[0] != '/') {
        return arrow::Status::Invalid("S3 URI should be absolute, not relative");
      }
      path = bucket + path;
    }
  }
  if (out_path != nullptr) {
    *out_path = std::string(arrow::fs::internal::RemoveTrailingSlash(path));
  }

  std::unordered_map<std::string, std::string> options_map;
  ARROW_ASSIGN_OR_RAISE(const auto options_items, uri.query_items());
  for (const auto& kv : options_items) {
    options_map.emplace(kv.first, kv.second);
  }

  const auto username = uri.username();
  if (!username.empty()) {
    options.ConfigureAccessKey(username, uri.password());
  } else {
    options.ConfigureDefaultCredentials();
  }
  // Prefer AWS service-specific endpoint url
  auto s3_endpoint_env = GetEnvVar(kAwsEndpointUrlS3EnvVar);
  if (s3_endpoint_env.ok()) {
    options.endpoint_override = *s3_endpoint_env;
  } else {
    auto endpoint_env = GetEnvVar(kAwsEndpointUrlEnvVar);
    if (endpoint_env.ok()) {
      options.endpoint_override = *endpoint_env;
    }
  }

  bool region_set = false;
  for (const auto& kv : options_map) {
    if (kv.first == "region") {
      options.region = kv.second;
      region_set = true;
    } else if (kv.first == "scheme") {
      options.scheme = kv.second;
    } else if (kv.first == "endpoint_override") {
      options.endpoint_override = kv.second;
    } else if (kv.first == "allow_bucket_creation") {
      ARROW_ASSIGN_OR_RAISE(options.allow_bucket_creation, ::arrow::internal::ParseBoolean(kv.second));
    } else if (kv.first == "allow_bucket_deletion") {
      ARROW_ASSIGN_OR_RAISE(options.allow_bucket_deletion, ::arrow::internal::ParseBoolean(kv.second));
    } else {
      return arrow::Status::Invalid("Unexpected query parameter in S3 URI: '", kv.first, "'");
    }
  }

  if (!region_set && !bucket.empty() && options.endpoint_override.empty()) {
    // XXX Should we use a dedicated resolver with the given credentials?
    ARROW_ASSIGN_OR_RAISE(options.region, ResolveS3BucketRegion(bucket));
  }

  return options;
}

arrow::Result<S3Options> S3Options::FromUri(const std::string& uri_string, std::string* out_path) {
  arrow::util::Uri uri;
  RETURN_NOT_OK(uri.Parse(uri_string));
  return FromUri(uri, out_path);
}

bool S3Options::Equals(const S3Options& other) const {
  const int64_t default_metadata_size = default_metadata ? default_metadata->size() : 0;
  const bool default_metadata_equals =
      default_metadata_size ? (other.default_metadata && other.default_metadata->Equals(*default_metadata))
                            : (!other.default_metadata || other.default_metadata->size() == 0);
  return (region == other.region && connect_timeout == other.connect_timeout &&
          request_timeout == other.request_timeout && endpoint_override == other.endpoint_override &&
          scheme == other.scheme && role_arn == other.role_arn && session_name == other.session_name &&
          external_id == other.external_id && load_frequency == other.load_frequency &&
          proxy_options.Equals(other.proxy_options) && credentials_kind == other.credentials_kind &&
          background_writes == other.background_writes && allow_bucket_creation == other.allow_bucket_creation &&
          allow_bucket_deletion == other.allow_bucket_deletion && default_metadata_equals &&
          GetAccessKey() == other.GetAccessKey() && GetSecretKey() == other.GetSecretKey() &&
          GetSessionToken() == other.GetSessionToken());
}

bool S3ProxyOptions::Equals(const S3ProxyOptions& other) const {
  return scheme == other.scheme && host == other.host && port == other.port && username == other.username &&
         password == other.password;
}

// -----------------------------------------------------------------------
// Top-level utility functions

arrow::Result<std::string> ResolveS3BucketRegion(const std::string& bucket) {
  RETURN_NOT_OK(CheckS3Initialized());

  if (bucket.empty() || bucket.find_first_of(kSep) != bucket.npos || arrow::fs::internal::IsLikelyUri(bucket)) {
    return arrow::Status::Invalid("Not a valid bucket name: '", bucket, "'");
  }

  // Region resolution requires a full S3 client to be functional.
  // For now, we return an error suggesting the user specify the region explicitly.
  // This is a simplification from Arrow's full implementation which includes a RegionResolver.
  return arrow::Status::Invalid(
      "Cannot automatically resolve S3 bucket region. Please specify the region explicitly "
      "in S3Options or via environment variables (AWS_REGION or AWS_DEFAULT_REGION).");
}

}  // namespace milvus_storage
