// Copyright 2026 Zilliz
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
#include <string>
#include <utility>

#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/http/HttpRequest.h>

namespace milvus_storage {

struct ArrowFileSystemConfig;

// GcpCredentialProvider abstracts the identity attached to GCP requests.
//
// A provider is looked up per-request by (endpoint, bucket) via
// GcpCredentialRegistry. The GCP HTTP factory calls AuthorizationHeader() at
// request creation to inject OAuth2 Bearer (IAM modes), and the HTTP delegator
// calls MaybeSignConditionalWrite() to re-sign conditional writes with
// GOOG4-HMAC-SHA256 (HMAC mode).
//
// Concrete implementations (VM IAM / Impersonation / HMAC) live entirely in
// the .cpp — callers only ever see this interface and the factory below.
class GcpCredentialProvider {
  public:
  virtual ~GcpCredentialProvider() = default;

  // Returns {header_name, header_value} for IAM/Impersonation modes, or
  // std::nullopt for HMAC (AWS SDK's SigV4 will sign the request itself).
  virtual std::optional<std::pair<std::string, std::string>> AuthorizationHeader() = 0;

  // Re-signs conditional writes using GOOG4-HMAC-SHA256 (HMAC mode only).
  // No-op for IAM/Impersonation modes.
  virtual arrow::Status MaybeSignConditionalWrite(const std::shared_ptr<Aws::Http::HttpRequest>& request) = 0;
};

// Build the appropriate provider from a filesystem config. Returns
// arrow::Status::Invalid when the config doesn't match any supported credential
// mode (no use_iam, and incomplete/missing HMAC AK/SK).
arrow::Result<std::shared_ptr<GcpCredentialProvider>> BuildGcpProviderFromConfig(const ArrowFileSystemConfig& config);

}  // namespace milvus_storage
