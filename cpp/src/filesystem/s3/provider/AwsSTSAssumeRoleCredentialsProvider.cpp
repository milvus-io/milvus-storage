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

#include "milvus-storage/filesystem/s3/provider/AwsSTSAssumeRoleCredentialsProvider.h"

#include <chrono>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include <aws/core/client/UserAgent.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/AssumeRoleRequest.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/s3/provider/AwsDefaultCredentialsProvider.h"

namespace milvus_storage {

namespace {

constexpr int kRefreshGraceSeconds = 60;
constexpr int kMinAssumeRoleDurationSeconds = 900;
constexpr int kMaxAssumeRoleDurationSeconds = 43200;
constexpr const char* kLogTag = "AwsSTSAssumeRoleCredentialsProvider";

bool IsOneOf(std::string_view value, std::initializer_list<std::string_view> candidates) {
  for (auto candidate : candidates) {
    if (value == candidate) {
      return true;
    }
  }
  return false;
}

arrow::Status ClassifyAwsStsFailure(const Aws::Client::AWSError<Aws::STS::STSErrors>& error) {
  const std::string exception_name(error.GetExceptionName().data(), error.GetExceptionName().size());
  const std::string message(error.GetMessage().data(), error.GetMessage().size());
  const auto context =
      "AWS STS AssumeRole failed: exception=" + (exception_name.empty() ? std::string("(none)") : exception_name) +
      " message=" + message;

  // AWS models throttling as HTTP 400 for some STS error spellings. Preserve
  // the service error name before falling back to HTTP classification.
  if (IsOneOf(exception_name,
              {"Throttling", "ThrottlingException", "TooManyRequestsException", "RequestLimitExceeded"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientThrottling, context);
  }
  if (IsOneOf(exception_name, {"RequestTimeout", "RequestTimeoutException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientTimeout, context);
  }
  if (IsOneOf(exception_name, {"IDPCommunicationError", "IDPCommunicationErrorException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, context);
  }
  if (IsOneOf(exception_name, {"InternalFailure", "InternalServerError", "ServiceUnavailable"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientService, context);
  }
  if (IsOneOf(exception_name, {"AccessDenied", "AccessDeniedException", "InvalidClientTokenId", "InvalidIdentityToken",
                               "ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageAccessDenied, context);
  }
  return ClassifyCredentialHttpFailure(error.GetResponseCode(), context);
}

}  // namespace

AwsSTSAssumeRoleCredentialsProvider::AwsSTSAssumeRoleCredentialsProvider(
    const Aws::String& role_arn,
    const Aws::String& session_name,
    const Aws::String& external_id,
    int load_frequency,
    std::shared_ptr<Aws::STS::STSClient> sts_client,
    std::shared_ptr<RequestCredentialsResolver> source_resolver)
    : sts_client_(std::move(sts_client)),
      source_resolver_(std::move(source_resolver)),
      role_arn_(role_arn),
      session_name_(session_name),
      external_id_(external_id),
      load_frequency_(load_frequency) {
  if (sts_client_ == nullptr) {
    auto source_provider = std::make_shared<AwsDefaultCredentialsProvider>();
    source_resolver_ = source_provider;
    Aws::Client::ClientConfiguration client_config(
        Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
    client_config.connectTimeoutMs = kCredentialConnectTimeoutMs;
    client_config.requestTimeoutMs = kCredentialRequestTimeoutMs;
    client_config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>(kLogTag, kCredentialRetryAttempts);
    sts_client_ = Aws::MakeShared<Aws::STS::STSClient>(kLogTag, source_provider, client_config);
  }
  if (session_name_.empty()) {
    Aws::StringStream generated;
    generated << "milvus-storage-" << Aws::Utils::DateTime::CurrentTimeMillis();
    session_name_ = generated.str();
  }
  if (role_arn_.empty()) {
    last_resolution_ = MakeCredentialConfigError("AWS AssumeRole role ARN is empty");
  } else if (load_frequency_ < kMinAssumeRoleDurationSeconds || load_frequency_ > kMaxAssumeRoleDurationSeconds) {
    last_resolution_ = MakeCredentialConfigError("AWS AssumeRole duration must be in [900, 43200] seconds");
  }
}

Aws::Auth::AWSCredentials AwsSTSAssumeRoleCredentialsProvider::GetAWSCredentials() {
  auto result = ResolveForRequest();
  if (!result.ok()) {
    return {};
  }
  return std::move(result).ValueOrDie();
}

arrow::Result<Aws::Auth::AWSCredentials> AwsSTSAssumeRoleCredentialsProvider::ResolveForRequest() {
  const auto started = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!last_resolution_.ok() && credentials_.IsEmpty() &&
      (role_arn_.empty() || load_frequency_ < kMinAssumeRoleDurationSeconds ||
       load_frequency_ > kMaxAssumeRoleDurationSeconds)) {
    return last_resolution_;
  }

  if (RefreshRequiredLocked()) {
    if (!CredentialAttemptStillWorthMaking(started) && !last_resolution_.ok()) {
      return last_resolution_;
    }
    ReloadLocked();
    // This request needed a refresh. Do not hide its failure behind an
    // about-to-expire cached value: the S3 holder must stop the object request
    // and return the typed STS cause.
    if (!last_resolution_.ok()) {
      return last_resolution_;
    }
  }

  auto validation = ValidateTemporaryCredentials(credentials_, "AWS STS AssumeRole");
  if (validation.ok()) {
    return credentials_;
  }
  return last_resolution_.ok() ? validation : last_resolution_;
}

bool AwsSTSAssumeRoleCredentialsProvider::RefreshRequiredLocked() const {
  if (!ValidateTemporaryCredentials(credentials_, "AWS STS AssumeRole cache").ok()) {
    return true;
  }
  return credentials_.GetExpiration() - Aws::Utils::DateTime::Now() < std::chrono::seconds(kRefreshGraceSeconds);
}

void AwsSTSAssumeRoleCredentialsProvider::ReloadLocked() {
  if (source_resolver_ != nullptr) {
    auto source = source_resolver_->ResolveForRequest();
    if (!source.ok()) {
      last_resolution_ = source.status();
      LOG_STORAGE_ERROR_ << "AWS source credential refresh failed before target AssumeRole: "
                         << last_resolution_.ToString();
      return;
    }
  }

  Aws::STS::Model::AssumeRoleRequest request;
  request.WithRoleArn(role_arn_).WithRoleSessionName(session_name_).WithDurationSeconds(load_frequency_);
  if (!external_id_.empty()) {
    request.SetExternalId(external_id_);
  }

  auto outcome = sts_client_->AssumeRole(request);
  if (!outcome.IsSuccess()) {
    last_resolution_ = ClassifyAwsStsFailure(outcome.GetError());
    LOG_STORAGE_ERROR_ << "AWS AssumeRole refresh failed: " << last_resolution_.ToString();
    return;
  }

  const auto& sts_credentials = outcome.GetResult().GetCredentials();
  Aws::Auth::AWSCredentials candidate(sts_credentials.GetAccessKeyId(), sts_credentials.GetSecretAccessKey(),
                                      sts_credentials.GetSessionToken(), sts_credentials.GetExpiration());
  auto validation = ValidateTemporaryCredentials(candidate, "AWS STS AssumeRole");
  if (!validation.ok()) {
    last_resolution_ = validation;
    return;
  }

  candidate.AddUserAgentFeature(Aws::Client::UserAgentFeature::CREDENTIALS_STS_ASSUME_ROLE);
  credentials_ = std::move(candidate);
  last_resolution_ = arrow::Status::OK();
}

}  // namespace milvus_storage
