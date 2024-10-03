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
#include <string>
#include <vector>
#include <cstdlib>
#include "common/log.h"
#include "common/macro.h"

#include <arrow/util/key_value_metadata.h>
#include <arrow/filesystem/s3fs.h>
#include "arrow/filesystem/filesystem.h"
#include "arrow/util/macros.h"
#include "arrow/util/uri.h"

using ::arrow::Result;

namespace milvus_storage {

struct S3Options {
  int64_t part_upload_size = 10 * 1024 * 1024;  // 10MB
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
  arrow::fs::S3ProxyOptions proxy_options;

  /// AWS credentials provider
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider;

  /// Type of credentials being used. Set along with credentials_provider.
  arrow::fs::S3CredentialsKind credentials_kind = arrow::fs::S3CredentialsKind::Default;

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

  /// Whether to allow file-open methods to return before the actual open.
  ///
  /// Enabling this may reduce the latency of `OpenInputStream`, `OpenOutputStream`,
  /// and similar methods, by reducing the number of roundtrips necessary. It may also
  /// allow usage of more efficient S3 APIs for small files.
  /// The downside is that failure conditions such as attempting to open a file in a
  /// non-existing bucket will only be reported when actual I/O is done (at worse,
  /// when attempting to close the file).
  bool allow_delayed_open = false;

  /// \brief Default metadata for OpenOutputStream.
  ///
  /// This will be ignored if non-empty metadata is passed to OpenOutputStream.
  std::shared_ptr<const arrow::KeyValueMetadata> default_metadata;

  /// Optional retry strategy to determine which error types should be retried, and the
  /// delay between retries.
  std::shared_ptr<arrow::fs::S3RetryStrategy> retry_strategy;

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

  static Result<S3Options> FromUri(const ::arrow::util::Uri& uri, std::string* out_path = NULLPTR);
  static Result<S3Options> FromUri(const std::string& uri, std::string* out_path = NULLPTR);
};

class MultiPartUploadS3FS : public arrow::fs::S3FileSystem {
  public:
  explicit MultiPartUploadS3FS(const arrow::fs::S3Options& options, int64_t part_size)
      : options_(options), part_size_(part_size) {}

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  static Result<std::shared_ptr<S3FileSystem>> Make(const S3Options& options,
                                                    const io::IOContext& = io::default_io_context()) override;

  protected:
  class Impl;
  std::shared_ptr<Impl> impl_;

  private:
  const int64_t part_size_;
  const arrow::fs::S3Options& options_;
};

class MultiPartUploadS3FSProducer : public FileSystemProducer {
  public:
  MultiPartUploadS3FSProducer() {};

  Result<std::shared_ptr<arrow::fs::FileSystem>> Make(const std::string& uri, std::string* out_path) override {
    arrow::util::Uri uri_parser;
    RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));

    if (!arrow::fs::IsS3Initialized()) {
      arrow::fs::S3GlobalOptions global_options;
      RETURN_ARROW_NOT_OK(arrow::fs::InitializeS3(global_options));
      std::atexit([]() {
        auto status = arrow::fs::EnsureS3Finalized();
        if (!status.ok()) {
          LOG_STORAGE_WARNING_ << "Failed to finalize S3: " << status.message();
        }
      });
    }

    arrow::fs::S3Options options;
    options.endpoint_override = uri_parser.ToString();
    options.ConfigureAccessKey(std::getenv("ACCESS_KEY"), std::getenv("SECRET_KEY"));

    if (std::getenv("REGION") != nullptr) {
      options.region = std::getenv("REGION");
    }
    // TODO: move all env variables into config interface
    int64_t part_size = std::stoll(std::getenv("PART_SIZE")) * 1024 * 1024;

    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, MultiPartUploadS3FS::Make(options, part_size));
    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }
};

}  // namespace milvus_storage