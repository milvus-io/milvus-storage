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

#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/gcp/gcp_credential_provider.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

namespace milvus_storage {

// GCS access through the S3-compatible XML API, using the AWS SDK S3Client.
// Authentication is layered via a custom HttpClientFactory:
//   - IAM mode: OAuth2 Bearer token from ComputeEngineCredentials (GCE metadata).
//     The AWS SDK is fed anonymous credentials so it does not overwrite the
//     injected Authorization header.
//   - HMAC mode: GCS HMAC ak/sk with standard AWS SigV4 for non-conditional
//     requests, and GOOG4-HMAC-SHA256 re-signing for conditional writes
//     (x-goog-if-generation-match), which GCS does not accept via SigV4.
// The factory binds the SDK clients that Make() builds to that filesystem's
// provider, and S3Options::request_authenticator carries it to CRT meta
// requests and native transport requests. Requests keep the filesystem's
// identity when another filesystem later registers its (endpoint, bucket) in
// GcpCredentialRegistry.
class GcpFileSystemProducer : public FileSystemProducer {
  public:
  explicit GcpFileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  arrow::Result<ArrowFileSystemPtr> Make() override;

  private:
  friend class S3RequestTimeoutTest;
  friend class GcpClientBindingTest;

  // One-shot: install the stateless HTTP client factory and call
  // Aws::InitializeS3. Guarded by std::call_once; the first GCP Make() wins
  // and determines process-global settings (TLS floor, log level). Returns
  // the status captured the first time call_once actually ran.
  static arrow::Status InitS3Compat(const ArrowFileSystemConfig& first_config);

  // Registers provider for (address, bucket) and binds the SDK clients this
  // call builds to the registered provider, which is an existing provider
  // when the identity is the same. When the bucket already maps to another
  // identity, the filesystem uses provider without an entry if its clients
  // are bound, and fails otherwise. Make() passes the provider built from the
  // config.
  arrow::Result<ArrowFileSystemPtr> MakeWithProvider(std::shared_ptr<GcpCredentialProvider> provider);

  // bearer_token: the provider injects Authorization, so the AWS signer must
  // not sign (anonymous credentials).
  arrow::Result<S3Options> CreateS3Options(bool bearer_token);

  const ArrowFileSystemConfig config_;
};

}  // namespace milvus_storage
