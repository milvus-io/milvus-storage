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

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/sts/STSClient.h>

#include <arrow/util/key_value_metadata.h>
#include <arrow/result.h>
#include <arrow/util/uri.h>

#include "milvus-storage/filesystem/s3/s3_internal.h"

namespace milvus_storage {

class S3RetryStrategy {
  public:
  virtual ~S3RetryStrategy() = default;

  /// Simple struct where each field corresponds to a field in Aws::Client::AWSError
  struct AWSErrorDetail {
    /// Corresponds to AWSError::GetErrorType()
    int error_type;
    /// Corresponds to AWSError::GetMessage()
    std::string message;
    /// Corresponds to AWSError::GetExceptionName()
    std::string exception_name;
    /// Corresponds to AWSError::ShouldRetry()
    bool should_retry;
  };
  /// Returns true if the S3 request resulting in the provided error should be retried.
  virtual bool ShouldRetry(const AWSErrorDetail& error, int64_t attempted_retries) = 0;
  /// Returns the time in milliseconds the S3 client should sleep for until retrying.
  virtual int64_t CalculateDelayBeforeNextRetry(const AWSErrorDetail& error, int64_t attempted_retries) = 0;
  /// Returns a stock AWS Default retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsDefaultRetryStrategy(int64_t max_attempts);
  /// Returns a stock AWS Standard retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsStandardRetryStrategy(int64_t max_attempts);
};

/// Options for using a proxy for S3
struct S3ProxyOptions {
  std::string scheme;
  std::string host;
  int port = -1;
  std::string username;
  std::string password;

  /// Initialize from URI such as http://username:password@host:port
  /// or http://host:port
  static arrow::Result<S3ProxyOptions> FromUri(const std::string& uri);
  static arrow::Result<S3ProxyOptions> FromUri(const ::arrow::util::Uri& uri);

  bool Equals(const S3ProxyOptions& other) const;
};

enum class S3CredentialsKind : int8_t {
  /// Anonymous access (no credentials used)
  Anonymous,
  /// Use default AWS credentials, configured through environment variables
  Default,
  /// Use explicitly-provided access key pair
  Explicit,
  /// Assume role through a role ARN
  Role,
  /// Use web identity token to assume role, configured through environment variables
  WebIdentity
};

/// Options for the S3FileSystem implementation.
struct S3Options {
  /// \brief AWS region to connect to.
  ///
  /// If unset, the AWS SDK will choose a default value.  The exact algorithm
  /// depends on the SDK version.  Before 1.8, the default is hardcoded
  /// to "us-east-1".  Since 1.8, several heuristics are used to determine
  /// the region (environment variables, configuration profile, EC2 metadata
  /// server).
  std::string region;

  /// \brief Socket connection timeout, in seconds
  ///
  /// If negative, the AWS SDK default value is used (typically 1 second).
  double connect_timeout = -1;

  /// \brief Socket read timeout on Windows and macOS, in seconds
  ///
  /// If negative, the AWS SDK default value is used (typically 3 seconds).
  /// This option is ignored on non-Windows, non-macOS systems.
  double request_timeout = -1;

  /// If non-empty, override region with a connect string such as "localhost:9000"
  // XXX perhaps instead take a URL like "http://localhost:9000"?
  std::string endpoint_override;
  /// S3 connection transport, default "https"
  std::string scheme = "https";

  /// ARN of role to assume
  std::string role_arn;
  /// Optional identifier for an assumed role session.
  std::string session_name;
  /// Optional external identifier to pass to STS when assuming a role
  std::string external_id;
  /// Frequency (in seconds) to refresh temporary credentials from assumed role
  int load_frequency = 900;

  /// If connection is through a proxy, set options here
  S3ProxyOptions proxy_options;

  /// AWS credentials provider
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider;

  /// Type of credentials being used. Set along with credentials_provider.
  S3CredentialsKind credentials_kind = S3CredentialsKind::Default;

  /// Whether to use virtual addressing of buckets
  ///
  /// If true, then virtual addressing is always enabled.
  /// If false, then virtual addressing is only enabled if `endpoint_override` is empty.
  ///
  /// This can be used for non-AWS backends that only support virtual hosted-style access.
  bool force_virtual_addressing = false;

  /// Whether OutputStream writes will be issued in the background, without blocking.
  bool background_writes = true;

  /// Whether to allow creation of buckets
  ///
  /// When S3FileSystem creates new buckets, it does not pass any non-default settings.
  /// In AWS S3, the bucket and all objects will be not publicly visible, and there
  /// will be no bucket policies and no resource tags. To have more control over how
  /// buckets are created, use a different API to create them.
  bool allow_bucket_creation = false;

  /// Whether to allow deletion of buckets
  bool allow_bucket_deletion = false;

  /// Whether to allow pessimistic directory creation in CreateDir function
  ///
  /// By default, CreateDir function will try to create the directory without checking its
  /// existence. It's an optimization to try directory creation and catch the error,
  /// rather than issue two dependent I/O calls.
  /// Though for key/value storage like Google Cloud Storage, too many creation calls will
  /// breach the rate limit for object mutation operations and cause serious consequences.
  /// It's also possible you don't have creation access for the parent directory. Set it
  /// to be true to address these scenarios.
  bool check_directory_existence_before_creation = false;

  /// \brief Default metadata for OpenOutputStream.
  ///
  /// This will be ignored if non-empty metadata is passed to OpenOutputStream.
  std::shared_ptr<const arrow::KeyValueMetadata> default_metadata = nullptr;

  /// Optional retry strategy to determine which error types should be retried, and the
  /// delay between retries.
  std::shared_ptr<S3RetryStrategy> retry_strategy = nullptr;

  /// \brief Maximum number of connections to the S3 server
  uint32_t max_connections = 100;

  /// Cloud provider name, e.g., "aws", "minio", "google", "azure", "aliyun", "tencent"
  std::string cloud_provider;

  S3Options();

  /// Configure with the default AWS credentials provider chain.
  void ConfigureDefaultCredentials();

  /// Configure with anonymous credentials.  This will only let you access public buckets.
  void ConfigureAnonymousCredentials();

  /// Configure with explicit access and secret key.
  void ConfigureAccessKey(const std::string& access_key,
                          const std::string& secret_key,
                          const std::string& session_token = "");

  /// Configure with credentials from an assumed role.
  void ConfigureAssumeRoleCredentials(const std::string& role_arn,
                                      const std::string& session_name = "",
                                      const std::string& external_id = "",
                                      int load_frequency = 900,
                                      const std::shared_ptr<Aws::STS::STSClient>& stsClient = NULLPTR);

  /// Configure with credentials from role assumed using a web identity token
  void ConfigureAssumeRoleWithWebIdentityCredentials();

  std::string GetAccessKey() const;
  std::string GetSecretKey() const;
  std::string GetSessionToken() const;

  bool Equals(const S3Options& other) const;

  /// \brief Initialize with default credentials provider chain
  ///
  /// This is recommended if you use the standard AWS environment variables
  /// and/or configuration file.
  static S3Options Defaults();

  /// \brief Initialize with anonymous credentials.
  ///
  /// This will only let you access public buckets.
  static S3Options Anonymous();

  /// \brief Initialize with explicit access and secret key.
  ///
  /// Optionally, a session token may also be provided for temporary credentials
  /// (from STS).
  static S3Options FromAccessKey(const std::string& access_key,
                                 const std::string& secret_key,
                                 const std::string& session_token = "");

  /// \brief Initialize from an assumed role.
  static S3Options FromAssumeRole(const std::string& role_arn,
                                  const std::string& session_name = "",
                                  const std::string& external_id = "",
                                  int load_frequency = 900,
                                  const std::shared_ptr<Aws::STS::STSClient>& stsClient = NULLPTR);

  /// \brief Initialize from an assumed role with web-identity.
  /// Uses the AWS SDK which uses environment variables to
  /// generate temporary credentials.
  static S3Options FromAssumeRoleWithWebIdentity();

  static arrow::Result<S3Options> FromUri(const ::arrow::util::Uri& uri, std::string* out_path = NULLPTR);
  static arrow::Result<S3Options> FromUri(const std::string& uri, std::string* out_path = NULLPTR);
};

/// \brief Resolve the AWS region for a given S3 bucket
///
/// This function attempts to determine the region where an S3 bucket is located.
/// Note: This is a simplified implementation that requires the region to be
/// specified explicitly via S3Options or environment variables.
arrow::Result<std::string> ResolveS3BucketRegion(const std::string& bucket);

}  // namespace milvus_storage
