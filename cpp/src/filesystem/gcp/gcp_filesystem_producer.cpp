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

#include "milvus-storage/filesystem/gcp/gcp_filesystem_producer.h"

#include <arrow/filesystem/filesystem.h>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"

namespace milvus_storage {

arrow::Result<S3Options> GcpFileSystemProducer::CreateS3Options() {
  S3Options options;

  // Three cases:
  // 1. no ssl, verifySSL=false
  // 2. self-signed certificate, verifySSL=false
  // 3. CA-signed certificate, verifySSL=true
  options.scheme = config_.use_ssl ? "https" : "http";

  if (config_.use_ssl && !config_.ssl_ca_cert.empty()) {
    arrow::fs::FileSystemGlobalOptions fs_global_options;
    fs_global_options.tls_ca_file_path = config_.ssl_ca_cert;
    ARROW_RETURN_NOT_OK(arrow::fs::Initialize(fs_global_options));
  }

  options.endpoint_override = config_.address;
  options.force_virtual_addressing = config_.use_virtual_host;

  if (!config_.region.empty()) {
    options.region = config_.region;
  }

  // request_timeout is in fractional seconds (double). Divide by 1000.0, not
  // 1000: integer division truncates sub-second timeouts (1500ms -> 1s) and
  // zeroes any value below 1s, which the client builder then treats as "no
  // timeout". s3_client_builder applies ceil(request_timeout * 1000) to recover
  // the exact millisecond value.
  options.request_timeout = config_.request_timeout_ms <= 0 ? DEFAULT_ARROW_FILESYSTEM_S3_REQUEST_TIMEOUT_SEC
                                                            : config_.request_timeout_ms / 1000.0;
  options.max_connections = config_.max_connections;
  options.crt_queue_sizes = config_.crt_queue_sizes;
  options.multi_part_upload_size = config_.multi_part_upload_size;
  options.cloud_provider = config_.cloud_provider;
  options.background_writes = config_.background_writes;
  options.use_crc32c_checksum = config_.use_crc32c_checksum;
  options.use_crt = config_.s3_use_crt;

  // GCP does not support AssumeRole
  if (!config_.role_arn.empty()) {
    return arrow::Status::Invalid("AssumeRole credentials are only supported for AWS cloud provider, got: ",
                                  config_.cloud_provider);
  }

  ARROW_ASSIGN_OR_RAISE(options.credentials_provider, BuildGcpProviderFromConfig(config_));
  options.credentials_kind = options.credentials_provider->GetAWSCredentials().IsEmpty() ? S3CredentialsKind::Anonymous
                                                                                         : S3CredentialsKind::Explicit;
  return options;
}

arrow::Result<ArrowFileSystemPtr> GcpFileSystemProducer::Make() {
  ARROW_RETURN_NOT_OK(S3FileSystemProducer(config_).InitS3());
  ARROW_ASSIGN_OR_RAISE(auto options, CreateS3Options());
  if (config_.alias.empty()) {
    GcpCredentialProvider::TrustEndpoint(config_);
  }
  return S3FileSystem::Make(options);
}

}  // namespace milvus_storage
