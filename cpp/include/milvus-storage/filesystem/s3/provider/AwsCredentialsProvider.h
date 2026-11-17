// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/client/ClientConfiguration.h>

namespace Aws::Crt::Auth {
class ICredentialsProvider;
}
namespace Aws::STS {
class STSClient;
}

namespace milvus_storage {

// Protect one credential source, never an entire identity-selection chain.
// Temporary sources must provide a complete STS tuple and a real expiration.
// Static profile credentials may explicitly opt into an unlimited lifetime.
class ValidatedAwsCredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
  public:
  using Factory = std::function<std::shared_ptr<Aws::Auth::AWSCredentialsProvider>()>;
  using Clock = std::function<Aws::Utils::DateTime()>;

  explicit ValidatedAwsCredentialsProvider(Factory factory,
                                           bool allow_static_credentials = false,
                                           Clock clock = {},
                                           std::chrono::milliseconds refresh_grace = std::chrono::minutes(5));
  Aws::Auth::AWSCredentials GetAWSCredentials() override;

  private:
  bool IsUsable(const Aws::Auth::AWSCredentials& credentials, const Aws::Utils::DateTime& now) const;
  Factory factory_;
  const bool allow_static_credentials_;
  Clock clock_;
  const std::chrono::milliseconds refresh_grace_;
  std::mutex mutex_;
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> delegate_;
  Aws::Auth::AWSCredentials credentials_;
};

// Bridge a single CRT source without discarding its expiration. The SDK's
// InstanceProfileCredentialsProvider does not propagate that expiration.
std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsCrtCredentialsProvider(
    std::shared_ptr<Aws::Crt::Auth::ICredentialsProvider> source,
    std::chrono::milliseconds timeout = std::chrono::seconds(10),
    ValidatedAwsCredentialsProvider::Clock clock = {});

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsWebIdentityCredentialsProvider();
std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsInstanceProfileCredentialsProvider();

// Preserve the SDK's configured source order and eligibility. Each refreshable
// source has an independent cache, so fallback never imports another identity's
// tuple into the failed source's cache.
std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsDefaultCredentialsProviderChain(
    const Aws::Client::ClientConfiguration::CredentialProviderConfiguration& config);
std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsDefaultCredentialsProviderChain();

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsAssumeRoleCredentialsProvider(
    const Aws::String& role_arn,
    const Aws::String& session_name,
    const Aws::String& external_id,
    int duration_seconds,
    std::shared_ptr<Aws::STS::STSClient> client = nullptr,
    ValidatedAwsCredentialsProvider::Clock clock = {});

}  // namespace milvus_storage
