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

#include "milvus-storage/filesystem/s3/provider/AwsDefaultCredentialsProvider.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <aws/core/auth/SSOCredentialsProvider.h>
#include <aws/core/client/UserAgent.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/config/ConfigAndCredentialsCacheManager.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/http/URI.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/AssumeRoleWithWebIdentityRequest.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/log.h"

namespace milvus_storage {

namespace {

constexpr const char* kLogTag = "AwsDefaultCredentialsProvider";
constexpr const char* kDefaultImdsEndpoint = "http://169.254.169.254";
constexpr const char* kIpv6ImdsEndpoint = "http://[fd00:ec2::254]";
constexpr const char* kImdsTokenPath = "/latest/api/token";
constexpr const char* kImdsRolePath = "/latest/meta-data/iam/security-credentials/";
constexpr const char* kImdsTokenTtlHeader = "x-aws-ec2-metadata-token-ttl-seconds";
constexpr const char* kImdsTokenHeader = "x-aws-ec2-metadata-token";
constexpr const char* kImdsTokenTtl = "21600";
constexpr auto kRefreshGrace = std::chrono::seconds(60);

bool IsTrue(Aws::String value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value == "true";
}

bool IsOneOf(std::string_view value, std::initializer_list<std::string_view> candidates) {
  for (auto candidate : candidates) {
    if (value == candidate) {
      return true;
    }
  }
  return false;
}

arrow::Result<std::optional<Aws::Auth::AWSCredentials>> ValidateSdkProviderCredentials(
    Aws::Auth::AWSCredentials credentials, std::string_view source) {
  const bool has_any_field = !credentials.GetAWSAccessKeyId().empty() || !credentials.GetAWSSecretKey().empty() ||
                             !credentials.GetSessionToken().empty();
  const bool has_expiration =
      credentials.GetExpiration().UnderlyingTimestamp() != (std::chrono::system_clock::time_point::max)();
  if (!has_any_field && !has_expiration) {
    return std::nullopt;
  }

  if (has_expiration) {
    ARROW_RETURN_NOT_OK(ValidateTemporaryCredentials(credentials, source));
  } else if (credentials.GetAWSAccessKeyId().empty() || credentials.GetAWSSecretKey().empty()) {
    return MakeCredentialResponseError(std::string(source) + " returned incomplete non-expiring credentials");
  }
  return std::optional<Aws::Auth::AWSCredentials>(std::move(credentials));
}

std::string ReadBody(const std::shared_ptr<Aws::Http::HttpResponse>& response) {
  if (response == nullptr) {
    return {};
  }
  std::ostringstream body;
  body << response->GetResponseBody().rdbuf();
  return body.str();
}

std::string Trim(std::string value) { return std::string(Aws::Utils::StringUtils::Trim(value.c_str()).c_str()); }

Aws::Config::Profile CurrentProfile() { return Aws::Config::GetCachedConfigProfile(Aws::Auth::GetConfigProfileName()); }

Aws::String EnvOrProfile(const char* env_name, const Aws::String& profile_value) {
  auto value = Aws::Environment::GetEnv(env_name);
  return value.empty() ? profile_value : value;
}

struct WebIdentityConfig {
  Aws::String role_arn;
  Aws::String token_file;
  Aws::String session_name;
  Aws::String region;

  bool Declared() const { return !role_arn.empty() || !token_file.empty(); }
};

WebIdentityConfig GetWebIdentityConfig() {
  const auto profile = CurrentProfile();
  WebIdentityConfig config;
  config.role_arn = EnvOrProfile("AWS_ROLE_ARN", profile.GetRoleArn());
  config.token_file = EnvOrProfile("AWS_WEB_IDENTITY_TOKEN_FILE", profile.GetValue("web_identity_token_file"));
  config.session_name = EnvOrProfile("AWS_ROLE_SESSION_NAME", profile.GetValue("role_session_name"));
  config.region = Aws::Environment::GetEnv("AWS_REGION");
  if (config.region.empty()) {
    config.region = EnvOrProfile("AWS_DEFAULT_REGION", profile.GetRegion());
  }
  if (config.region.empty()) {
    // STS supports this region in every standard partition and using an
    // explicit value prevents ClientConfiguration from probing IMDS just to
    // discover an endpoint before credentials can be resolved.
    config.region = "us-east-1";
  }
  return config;
}

arrow::Status ClassifyAwsStsFailure(const Aws::Client::AWSError<Aws::STS::STSErrors>& error,
                                    std::string_view operation) {
  const std::string exception(error.GetExceptionName().data(), error.GetExceptionName().size());
  const std::string message(error.GetMessage().data(), error.GetMessage().size());
  const auto context = std::string(operation) +
                       " failed: exception=" + (exception.empty() ? std::string("(none)") : exception) +
                       " message=" + message;
  if (IsOneOf(exception, {"Throttling", "ThrottlingException", "TooManyRequestsException", "RequestLimitExceeded"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientThrottling, context);
  }
  if (IsOneOf(exception, {"RequestTimeout", "RequestTimeoutException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientTimeout, context);
  }
  if (IsOneOf(exception, {"IDPCommunicationError", "IDPCommunicationErrorException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, context);
  }
  if (IsOneOf(exception, {"InternalFailure", "InternalServerError", "ServiceUnavailable"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientService, context);
  }
  if (IsOneOf(exception, {"AccessDenied", "AccessDeniedException", "InvalidClientTokenId", "InvalidIdentityToken",
                          "ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException"})) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageAccessDenied, context);
  }
  return ClassifyCredentialHttpFailure(error.GetResponseCode(), context);
}

std::shared_ptr<Aws::Http::HttpRequest> MakeMetadataRequest(const std::string& url, Aws::Http::HttpMethod method) {
  auto request = Aws::Http::CreateHttpRequest(Aws::String(url.c_str(), url.size()), method,
                                              Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  request->SetUserAgent(Aws::Client::ComputeUserAgentString());
  return request;
}

bool ImdsV1FallbackAllowed(Aws::Http::HttpResponseCode code) {
  using Aws::Http::HttpResponseCode;
  return code == HttpResponseCode::FORBIDDEN || code == HttpResponseCode::NOT_FOUND ||
         code == HttpResponseCode::METHOD_NOT_ALLOWED;
}

bool IsAllowedContainerHttpHost(const Aws::String& host) {
  if (host == "169.254.170.2" || host == "169.254.170.23" || host == "fd00:ec2::23" || host == "[fd00:ec2::23]" ||
      host == "::1" || host == "[::1]") {
    return true;
  }
  constexpr std::string_view loopback_prefix = "127.0.0.";
  if (host.rfind(loopback_prefix.data(), 0) != 0 || host.size() <= loopback_prefix.size()) {
    return false;
  }
  const auto suffix = host.substr(loopback_prefix.size());
  if (suffix.find_first_not_of("0123456789") != Aws::String::npos) {
    return false;
  }
  const auto octet = Aws::Utils::StringUtils::ConvertToInt32(suffix.c_str());
  return octet >= 0 && octet <= 255;
}

arrow::Result<std::string> ContainerCredentialUrl() {
  const auto relative = Aws::Environment::GetEnv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  if (!relative.empty()) {
    if (relative.front() != '/') {
      return MakeCredentialConfigError("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI must begin with '/'");
    }
    return std::string("http://169.254.170.2") + relative.c_str();
  }

  const auto absolute = Aws::Environment::GetEnv("AWS_CONTAINER_CREDENTIALS_FULL_URI");
  if (absolute.empty()) {
    return MakeCredentialConfigError("AWS container credential URI is not configured");
  }
  const Aws::Http::URI uri(absolute);
  if (uri.GetScheme() != Aws::Http::Scheme::HTTPS &&
      !(uri.GetScheme() == Aws::Http::Scheme::HTTP && IsAllowedContainerHttpHost(uri.GetHost()))) {
    return MakeCredentialConfigError("AWS_CONTAINER_CREDENTIALS_FULL_URI must use HTTPS or an AWS/loopback HTTP host");
  }
  return std::string(absolute.c_str(), absolute.size());
}

arrow::Result<std::string> ContainerAuthorizationToken() {
  auto token = Aws::Environment::GetEnv("AWS_CONTAINER_AUTHORIZATION_TOKEN");
  const auto token_file = Aws::Environment::GetEnv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE");
  if (!token_file.empty()) {
    std::ifstream input(token_file.c_str(), std::ios::binary);
    if (!input.is_open()) {
      return MakeCredentialConfigError("AWS container authorization token file cannot be opened");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    token = contents.str().c_str();
  }
  if (token.find_first_of("\r\n") != Aws::String::npos) {
    return MakeCredentialConfigError("AWS container authorization token contains a newline");
  }
  return std::string(token.c_str(), token.size());
}

arrow::Result<std::string> ImdsEndpoint() {
  const auto profile = CurrentProfile();
  const auto configured =
      EnvOrProfile("AWS_EC2_METADATA_SERVICE_ENDPOINT", profile.GetValue("ec2_metadata_service_endpoint"));
  if (!configured.empty()) {
    return std::string(configured.c_str(), configured.size());
  }
  const auto mode = Aws::Utils::StringUtils::ToLower(
      EnvOrProfile("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE", profile.GetValue("ec2_metadata_service_endpoint_mode"))
          .c_str());
  if (mode.empty() || mode == "ipv4") {
    return kDefaultImdsEndpoint;
  }
  if (mode == "ipv6") {
    return kIpv6ImdsEndpoint;
  }
  return MakeCredentialConfigError("AWS EC2 metadata endpoint mode must be ipv4 or ipv6");
}

bool ImdsV1Disabled() {
  const auto profile = CurrentProfile();
  return IsTrue(EnvOrProfile("AWS_EC2_METADATA_V1_DISABLED", profile.GetValue("ec2_metadata_v1_disabled")));
}

}  // namespace

AwsDefaultCredentialsProvider::Dependencies AwsDefaultCredentialsProvider::MakeDefaultDependencies() {
  Dependencies dependencies;
  dependencies.before_web_identity = {
      Aws::MakeShared<Aws::Auth::EnvironmentAWSCredentialsProvider>(kLogTag),
      Aws::MakeShared<Aws::Auth::ProfileConfigFileAWSCredentialsProvider>(kLogTag),
      Aws::MakeShared<Aws::Auth::ProcessCredentialsProvider>(kLogTag),
  };
  dependencies.after_web_identity = {
      Aws::MakeShared<Aws::Auth::SSOCredentialsProvider>(kLogTag),
  };

  Aws::Client::ClientConfiguration metadata_config(
      Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  metadata_config.scheme = Aws::Http::Scheme::HTTP;
  metadata_config.connectTimeoutMs = kCredentialConnectTimeoutMs;
  metadata_config.requestTimeoutMs = kImdsRequestTimeoutMs;
  dependencies.metadata_client = Aws::Http::CreateHttpClient(metadata_config);
  return dependencies;
}

AwsDefaultCredentialsProvider::AwsDefaultCredentialsProvider()
    : AwsDefaultCredentialsProvider(MakeDefaultDependencies(), SourceMode::DefaultChain) {}

AwsDefaultCredentialsProvider::AwsDefaultCredentialsProvider(SourceMode source_mode)
    : AwsDefaultCredentialsProvider(MakeDefaultDependencies(), source_mode) {}

AwsDefaultCredentialsProvider::AwsDefaultCredentialsProvider(Dependencies dependencies,
                                                             SourceMode source_mode)
    : dependencies_(std::move(dependencies)), source_mode_(source_mode) {
  if (dependencies_.metadata_client == nullptr) {
    Aws::Client::ClientConfiguration metadata_config(
        Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
    metadata_config.scheme = Aws::Http::Scheme::HTTP;
    metadata_config.connectTimeoutMs = kCredentialConnectTimeoutMs;
    metadata_config.requestTimeoutMs = kImdsRequestTimeoutMs;
    dependencies_.metadata_client = Aws::Http::CreateHttpClient(metadata_config);
  }
}

Aws::Auth::AWSCredentials AwsDefaultCredentialsProvider::GetAWSCredentials() {
  auto result = ResolveForRequest();
  if (!result.ok()) {
    return {};
  }
  return std::move(result).ValueOrDie();
}

arrow::Result<Aws::Auth::AWSCredentials> AwsDefaultCredentialsProvider::ResolveForRequest() {
  std::lock_guard<std::mutex> lock(mutex_);

  const auto web_identity = GetWebIdentityConfig();
  if (source_mode_ == SourceMode::WebIdentityOnly) {
    if (web_identity.role_arn.empty() || web_identity.token_file.empty()) {
      return MakeCredentialConfigError(
          "AWS web identity requires both AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE (or profile equivalents)");
    }
    return ResolveDynamic(DynamicSource::WebIdentity);
  }

  for (const auto& provider : dependencies_.before_web_identity) {
    if (provider == nullptr) {
      continue;
    }
    auto credentials = provider->GetAWSCredentials();
    ARROW_ASSIGN_OR_RAISE(auto validated,
                          ValidateSdkProviderCredentials(std::move(credentials), "AWS SDK default-chain provider"));
    if (validated.has_value()) {
      return std::move(*validated);
    }
  }

  if (web_identity.Declared()) {
    if (web_identity.role_arn.empty() || web_identity.token_file.empty()) {
      return MakeCredentialConfigError(
          "AWS web identity requires both AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE (or profile equivalents)");
    }
    // A declared workload identity is authoritative. In particular, do not
    // turn an IRSA outage into the node role by falling through to IMDS.
    return ResolveDynamic(DynamicSource::WebIdentity);
  }

  for (const auto& provider : dependencies_.after_web_identity) {
    if (provider == nullptr) {
      continue;
    }
    auto credentials = provider->GetAWSCredentials();
    ARROW_ASSIGN_OR_RAISE(auto validated,
                          ValidateSdkProviderCredentials(std::move(credentials), "AWS SDK default-chain provider"));
    if (validated.has_value()) {
      return std::move(*validated);
    }
  }

  const bool container_declared = !Aws::Environment::GetEnv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").empty() ||
                                  !Aws::Environment::GetEnv("AWS_CONTAINER_CREDENTIALS_FULL_URI").empty();
  if (container_declared) {
    // ECS/EKS Pod Identity is also authoritative; never silently borrow the
    // node role when its endpoint is unavailable.
    return ResolveDynamic(DynamicSource::Container);
  }

  if (IsTrue(Aws::Environment::GetEnv("AWS_EC2_METADATA_DISABLED"))) {
    return MakeCredentialConfigError("AWS default credential chain found no credentials and IMDS is disabled");
  }
  return ResolveDynamic(DynamicSource::Imds);
}

bool AwsDefaultCredentialsProvider::CachedDynamicCredentialsAreFresh(DynamicSource source) const {
  if (dynamic_source_ != source || !ValidateTemporaryCredentials(dynamic_credentials_, "AWS dynamic cache").ok()) {
    return false;
  }
  return dynamic_credentials_.GetExpiration() - Aws::Utils::DateTime::Now() >= kRefreshGrace;
}

arrow::Result<Aws::Auth::AWSCredentials> AwsDefaultCredentialsProvider::ResolveDynamic(DynamicSource source) {
  if (CachedDynamicCredentialsAreFresh(source)) {
    return dynamic_credentials_;
  }

  arrow::Result<Aws::Auth::AWSCredentials> resolved = MakeCredentialResponseError("unknown AWS credential source");
  switch (source) {
    case DynamicSource::WebIdentity:
      resolved = ResolveWebIdentity();
      break;
    case DynamicSource::Container:
      resolved = ResolveContainerCredentials();
      break;
    case DynamicSource::Imds:
      resolved = ResolveImdsCredentials();
      break;
    case DynamicSource::None:
      break;
  }

  dynamic_source_ = source;
  if (!resolved.ok()) {
    last_dynamic_resolution_ = resolved.status();
    return last_dynamic_resolution_;
  }

  auto credentials = std::move(resolved).ValueOrDie();
  auto validation = ValidateTemporaryCredentials(credentials, "AWS default credential source");
  if (!validation.ok()) {
    last_dynamic_resolution_ = validation;
    return last_dynamic_resolution_;
  }
  dynamic_credentials_ = std::move(credentials);
  last_dynamic_resolution_ = arrow::Status::OK();
  return dynamic_credentials_;
}

arrow::Result<Aws::Auth::AWSCredentials> AwsDefaultCredentialsProvider::ResolveWebIdentity() {
  const auto config = GetWebIdentityConfig();
  std::ifstream token_file(config.token_file.c_str(), std::ios::binary);
  if (!token_file.is_open()) {
    return MakeCredentialConfigError("AWS web identity token file cannot be opened");
  }
  std::ostringstream token_contents;
  token_contents << token_file.rdbuf();
  const auto token = Trim(token_contents.str());
  if (token.empty()) {
    return MakeCredentialConfigError("AWS web identity token file is empty");
  }

  if (dependencies_.web_identity_client == nullptr) {
    Aws::Client::ClientConfiguration client_config(
        Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
    client_config.region = config.region;
    client_config.connectTimeoutMs = kCredentialConnectTimeoutMs;
    client_config.requestTimeoutMs = kCredentialRequestTimeoutMs;
    client_config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>(kLogTag, kCredentialRetryAttempts);
    dependencies_.web_identity_client =
        Aws::MakeShared<Aws::STS::STSClient>(kLogTag, Aws::Auth::AWSCredentials{}, client_config);
  }

  Aws::STS::Model::AssumeRoleWithWebIdentityRequest request;
  request.SetRoleArn(config.role_arn);
  request.SetWebIdentityToken(token.c_str());
  request.SetRoleSessionName(config.session_name.empty() ? "milvus-storage" : config.session_name);
  auto outcome = dependencies_.web_identity_client->AssumeRoleWithWebIdentity(request);
  if (!outcome.IsSuccess()) {
    return ClassifyAwsStsFailure(outcome.GetError(), "AWS STS AssumeRoleWithWebIdentity");
  }

  const auto& result = outcome.GetResult().GetCredentials();
  return Aws::Auth::AWSCredentials(result.GetAccessKeyId(), result.GetSecretAccessKey(), result.GetSessionToken(),
                                   result.GetExpiration());
}

arrow::Result<Aws::Auth::AWSCredentials> AwsDefaultCredentialsProvider::ResolveContainerCredentials() {
  if (dependencies_.metadata_client == nullptr) {
    return ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode::REQUEST_NOT_MADE,
                                         "AWS container credentials have no HTTP client");
  }
  ARROW_ASSIGN_OR_RAISE(auto url, ContainerCredentialUrl());
  ARROW_ASSIGN_OR_RAISE(auto authorization_token, ContainerAuthorizationToken());
  auto request = MakeMetadataRequest(url, Aws::Http::HttpMethod::HTTP_GET);
  if (!authorization_token.empty()) {
    request->SetHeaderValue("Authorization", authorization_token.c_str());
  }
  const auto response = MakeRequestWithCredentialRetry(*dependencies_.metadata_client, request);
  if (response == nullptr || response->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    return ClassifyCredentialHttpFailure(
        response == nullptr ? Aws::Http::HttpResponseCode::NO_RESPONSE : response->GetResponseCode(),
        "AWS container credential request failed");
  }

  Aws::Utils::Json::JsonValue json(ReadBody(response).c_str());
  if (!json.WasParseSuccessful()) {
    return MakeCredentialResponseError("AWS container credential response is not valid JSON");
  }
  const auto view = json.View();
  Aws::Auth::AWSCredentials credentials(
      view.GetString("AccessKeyId"), view.GetString("SecretAccessKey"), view.GetString("Token"),
      Aws::Utils::DateTime(view.GetString("Expiration"), Aws::Utils::DateFormat::ISO_8601));
  return credentials;
}

arrow::Result<Aws::Auth::AWSCredentials> AwsDefaultCredentialsProvider::ResolveImdsCredentials() {
  if (dependencies_.metadata_client == nullptr) {
    return ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode::REQUEST_NOT_MADE, "AWS IMDS has no HTTP client");
  }
  ARROW_ASSIGN_OR_RAISE(auto endpoint, ImdsEndpoint());
  auto token_request = MakeMetadataRequest(endpoint + kImdsTokenPath, Aws::Http::HttpMethod::HTTP_PUT);
  token_request->SetHeaderValue(kImdsTokenTtlHeader, kImdsTokenTtl);
  token_request->SetContentLength("0");
  const auto token_response = MakeRequestWithCredentialRetry(*dependencies_.metadata_client, token_request);

  std::string token;
  if (token_response != nullptr && token_response->GetResponseCode() == Aws::Http::HttpResponseCode::OK) {
    token = Trim(ReadBody(token_response));
    if (token.empty()) {
      return MakeCredentialResponseError("AWS IMDSv2 returned an empty token");
    }
  } else {
    const auto code =
        token_response == nullptr ? Aws::Http::HttpResponseCode::NO_RESPONSE : token_response->GetResponseCode();
    if (ImdsV1Disabled() || !ImdsV1FallbackAllowed(code)) {
      return ClassifyCredentialHttpFailure(code, "AWS IMDSv2 token request failed");
    }
  }

  auto role_request = MakeMetadataRequest(endpoint + kImdsRolePath, Aws::Http::HttpMethod::HTTP_GET);
  if (!token.empty()) {
    role_request->SetHeaderValue(kImdsTokenHeader, token.c_str());
  }
  const auto role_response = MakeRequestWithCredentialRetry(*dependencies_.metadata_client, role_request);
  if (role_response == nullptr || role_response->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    return ClassifyCredentialHttpFailure(
        role_response == nullptr ? Aws::Http::HttpResponseCode::NO_RESPONSE : role_response->GetResponseCode(),
        "AWS IMDS role-list request failed");
  }
  auto role_name = Trim(ReadBody(role_response));
  const auto newline = role_name.find_first_of("\r\n");
  if (newline != std::string::npos) {
    role_name.resize(newline);
  }
  if (role_name.empty()) {
    return MakeCredentialConfigError("AWS IMDS reports no instance profile role");
  }

  auto credentials_request = MakeMetadataRequest(endpoint + kImdsRolePath + role_name, Aws::Http::HttpMethod::HTTP_GET);
  if (!token.empty()) {
    credentials_request->SetHeaderValue(kImdsTokenHeader, token.c_str());
  }
  const auto credentials_response = MakeRequestWithCredentialRetry(*dependencies_.metadata_client, credentials_request);
  if (credentials_response == nullptr || credentials_response->GetResponseCode() != Aws::Http::HttpResponseCode::OK) {
    return ClassifyCredentialHttpFailure(credentials_response == nullptr ? Aws::Http::HttpResponseCode::NO_RESPONSE
                                                                         : credentials_response->GetResponseCode(),
                                         "AWS IMDS credential request failed");
  }

  Aws::Utils::Json::JsonValue json(ReadBody(credentials_response).c_str());
  if (!json.WasParseSuccessful()) {
    return MakeCredentialResponseError("AWS IMDS credential response is not valid JSON");
  }
  const auto view = json.View();
  if (view.KeyExists("Code") && view.GetString("Code") != "Success") {
    return MakeCredentialResponseError("AWS IMDS credential response reported a non-success Code");
  }
  return Aws::Auth::AWSCredentials(
      view.GetString("AccessKeyId"), view.GetString("SecretAccessKey"), view.GetString("Token"),
      Aws::Utils::DateTime(view.GetString("Expiration"), Aws::Utils::DateFormat::ISO_8601));
}

}  // namespace milvus_storage
