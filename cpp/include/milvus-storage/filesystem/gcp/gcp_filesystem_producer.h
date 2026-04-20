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
class GcpFileSystemProducer : public FileSystemProducer {
  public:
  explicit GcpFileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  arrow::Result<ArrowFileSystemPtr> Make() override;

  private:
  arrow::Status InitS3Compat();

  arrow::Result<S3Options> CreateS3Options();

  const ArrowFileSystemConfig config_;
};

}  // namespace milvus_storage
