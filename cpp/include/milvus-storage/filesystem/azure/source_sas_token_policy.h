// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optional>
#include <memory>
#include <azure/core/http/policies/policy.hpp>

#include "milvus-storage/source_read_context.h"
#include "milvus-storage/common/extend_status.h"

namespace milvus_storage::fs {

class SourceCredentialStatusDetail final : public ExtendStatusDetail {
  public:
  explicit SourceCredentialStatusDetail(LoonSourceCredentialStatus status);
  // Keep ExtendStatusDetail's type_id: generic C ABI/segcore projections use
  // that identity before dynamically unwrapping the registered error code.
  std::string ToString() const override;
  LoonSourceCredentialStatus status() const { return status_; }

  private:
  LoonSourceCredentialStatus status_;
};

arrow::Status SourceCredentialArrowStatus(LoonSourceCredentialStatus status);

// The SDK policy interface returns HTTP responses. Local failures use a marked,
// non-retryable response carrier; no request is sent to Azure. Decode it before
// ordinary Azure error classification, preserving the original Arrow category.
std::optional<arrow::Status> SourcePolicyResponseStatus(const Azure::Core::Http::RawResponse& response);

// Install as PerRetryPolicies on both Blob and DataLake clients. Clones share
// credential state only within this context, partitioned by root and operation.
class SourceSasTokenPolicy final : public Azure::Core::Http::Policies::HttpPolicy {
  public:
  explicit SourceSasTokenPolicy(SourceReadContextPtr context);
  ~SourceSasTokenPolicy() override;
  std::unique_ptr<Azure::Core::Http::Policies::HttpPolicy> Clone() const override;
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(Azure::Core::Http::Request& request,
                                                       Azure::Core::Http::Policies::NextHttpPolicy next_policy,
                                                       const Azure::Core::Context& context) const override;

  private:
  struct SharedState;
  arrow::Result<std::string> GetSasToken(const SourceReadTarget& target, const Azure::Core::Context& context) const;
  arrow::Status CheckState(const Azure::Core::Context& context) const;
  std::shared_ptr<SharedState> state_;
};

}  // namespace milvus_storage::fs
