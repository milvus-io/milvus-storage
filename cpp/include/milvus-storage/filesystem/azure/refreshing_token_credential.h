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

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <azure/core/credentials/credentials.hpp>

namespace milvus_storage::fs {

// Keeps only validated tokens for the original credential object. Cache keys
// include all identity context (tenant and scopes); no identity fallback.
class RefreshingAzureCredential final : public Azure::Core::Credentials::TokenCredential {
  public:
  using Clock = std::function<std::chrono::system_clock::time_point()>;
  explicit RefreshingAzureCredential(std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate,
                                     Clock clock = {});
  Azure::Core::Credentials::AccessToken GetToken(const Azure::Core::Credentials::TokenRequestContext& request,
                                                 const Azure::Core::Context& context) const override;

  private:
  using Key = std::pair<std::string, std::vector<std::string>>;
  struct Entry {
    Azure::Core::Credentials::AccessToken token;
    bool force_refresh = false;
  };
  std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate_;
  Clock clock_;
  mutable std::mutex mutex_;
  mutable std::map<Key, Entry> cache_;
};

std::shared_ptr<Azure::Core::Credentials::TokenCredential> MakeRefreshingAzureCredential(
    std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate);

// AzureFileSystem catches StorageException/RequestFailedException at its Arrow
// boundary. Adapt local credential failure to that family without inventing an
// HTTP response or retry classification. The raw guard above keeps the SDK's
// AuthenticationException contract for other consumers.
std::shared_ptr<Azure::Core::Credentials::TokenCredential> MakeAzureStorageCredential(
    std::shared_ptr<Azure::Core::Credentials::TokenCredential> delegate);

}  // namespace milvus_storage::fs
