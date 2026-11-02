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

  // Whether AuthorizationHeader() can return an OAuth2 Bearer token. HMAC
  // providers sign requests instead and never return a header.
  [[nodiscard]] virtual bool IssuesBearerToken() const = 0;

  // A stable description of the identity this provider authenticates as, for
  // example "iam", "impersonate:<sa>", "broker:<endpoint>|<sa>" or
  // "hmac:<access_key_id>". GcpCredentialRegistry refuses to map one
  // (endpoint, bucket) to two providers with different identities.
  [[nodiscard]] virtual std::string Identity() const = 0;
};

// Build the appropriate provider from a filesystem config, in this order:
//   1. credential_endpoint without use_iam: the credential Broker mints tokens
//      for gcp_target_service_account.
//   2. gcp_target_service_account: this process impersonates the target.
//   3. use_iam: this process's own identity (VM SA or ADC).
//   4. HMAC access_key_id/access_key_value.
// Returns arrow::Status::Invalid when the config matches none of them.
arrow::Result<std::shared_ptr<GcpCredentialProvider>> BuildGcpProviderFromConfig(const ArrowFileSystemConfig& config);

// With HMAC keys and a target or Broker endpoint in one config, the native
// filesystem keeps HMAC when Talon is enabled (the Broker fields describe
// Talon's identity) and rejects the config as ambiguous otherwise.

}  // namespace milvus_storage
