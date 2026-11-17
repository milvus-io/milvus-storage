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

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <google/cloud/internal/oauth2_credentials.h>

namespace milvus_storage {

// The delegate must be an uncached credential with a fixed identity. Validate
// before publishing, so a rejected SDK result cannot poison an inner cache.
class RefreshingGcpCredentials final : public google::cloud::oauth2_internal::Credentials {
  public:
  using TimePoint = std::chrono::system_clock::time_point;
  using Clock = std::function<TimePoint()>;
  explicit RefreshingGcpCredentials(std::shared_ptr<google::cloud::oauth2_internal::Credentials> delegate,
                                    Clock clock = {});
  google::cloud::StatusOr<google::cloud::AccessToken> GetToken(TimePoint) override;
  google::cloud::StatusOr<std::vector<std::uint8_t>> SignBlob(const absl::optional<std::string>& account,
                                                              const std::string& value) const override;
  std::string AccountEmail() const override;
  std::string KeyId() const override;
  google::cloud::StatusOr<std::string> universe_domain() const override;
  google::cloud::StatusOr<std::string> universe_domain(const google::cloud::Options& options) const override;
  google::cloud::StatusOr<std::string> project_id() const override;
  google::cloud::StatusOr<std::string> project_id(const google::cloud::Options& options) const override;

  private:
  std::shared_ptr<google::cloud::oauth2_internal::Credentials> delegate_;
  Clock clock_;
  std::mutex mutex_;
  google::cloud::AccessToken cached_;
};

std::shared_ptr<google::cloud::oauth2_internal::Credentials> MakeRefreshingGcpDefaultCredentials();
std::shared_ptr<google::cloud::oauth2_internal::Credentials> MakeRefreshingGcpImpersonatedCredentials(
    const std::string& target, int lifetime_seconds);
std::shared_ptr<google::cloud::oauth2_internal::Credentials> MakeRefreshingGcpServiceAccountCredentials(
    const std::string& json);

}  // namespace milvus_storage
