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

#include "milvus-storage/filesystem/azure/refreshing_token_credential.h"
#include "milvus-storage/filesystem/bearer_token.h"
#include <azure/storage/common/storage_exception.hpp>

namespace milvus_storage::fs {
namespace {
using Azure::Core::Credentials::AccessToken;
bool Usable(const AccessToken& token, std::chrono::system_clock::time_point now) {
  return IsValidBearerToken(token.Token) && token.ExpiresOn > Azure::DateTime(now);
}

class AzureStorageCredential final : public Azure::Core::Credentials::TokenCredential {
  public:
  explicit AzureStorageCredential(std::shared_ptr<TokenCredential> credential)
      : TokenCredential(credential->GetCredentialName()), credential_(std::move(credential)) {}

  AccessToken GetToken(const Azure::Core::Credentials::TokenRequestContext& request,
                       const Azure::Core::Context& context) const override {
    try {
      return credential_->GetToken(request, context);
    } catch (const Azure::Core::Credentials::AuthenticationException& error) {
      // RefreshingAzureCredential has already sanitized the SDK cause. This is
      // a local failure, so StatusCode remains None and RawResponse stays null.
      // Existing AzureFS read/write catches then return Arrow Status, including
      // on worker threads where an uncaught exception would terminate a process.
      throw Azure::Storage::StorageException(error.what());
    }
  }

  private:
  std::shared_ptr<TokenCredential> credential_;
};
}  // namespace

RefreshingAzureCredential::RefreshingAzureCredential(
    std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate, Clock clock)
    : TokenCredential(delegate->GetCredentialName()),
      delegate_(std::move(delegate)),
      clock_(clock ? std::move(clock) : Clock([] { return std::chrono::system_clock::now(); })) {}

AccessToken RefreshingAzureCredential::GetToken(const Azure::Core::Credentials::TokenRequestContext& request,
                                                const Azure::Core::Context& context) const {
  std::lock_guard lock(mutex_);
  auto& entry = cache_[{request.TenantId, request.Scopes}];
  const bool caller_forced = request.MinimumExpiration == Azure::DateTime::duration::max();
  auto child_request = request;
  if (entry.force_refresh) {
    // This is the SDK's own invalidation mechanism after HTTP 401. TokenCache
    // uses ExpiresOn - MinimumExpiration, avoiding now + duration overflow.
    child_request.MinimumExpiration = Azure::DateTime::duration::max();
  }
  try {
    auto candidate = delegate_->GetToken(child_request, context);
    if (Usable(candidate, clock_())) {
      entry.token = std::move(candidate);
      entry.force_refresh = false;
      return entry.token;
    }
  } catch (...) {
    // Preserve the validated old value for a transient SDK/network/parse
    // failure. Do not include SDK exception messages: they may contain tokens.
  }
  entry.force_refresh = true;
  if (!caller_forced && Usable(entry.token, clock_())) {
    return entry.token;
  }
  // A forced refresh follows a server rejection, so the caller must not get
  // the rejected token back, even if its timestamp is still in the future.
  if (caller_forced) {
    entry.token = {};
  }
  throw Azure::Core::Credentials::AuthenticationException("Azure credential refresh failed; no usable cached token");
}

std::shared_ptr<Azure::Core::Credentials::TokenCredential> MakeRefreshingAzureCredential(
    std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate) {
  return std::make_shared<RefreshingAzureCredential>(std::move(delegate));
}

std::shared_ptr<Azure::Core::Credentials::TokenCredential> MakeAzureStorageCredential(
    std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate) {
  return std::make_shared<AzureStorageCredential>(MakeRefreshingAzureCredential(std::move(delegate)));
}

}  // namespace milvus_storage::fs
