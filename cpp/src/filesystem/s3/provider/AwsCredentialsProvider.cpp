// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "milvus-storage/filesystem/s3/provider/AwsCredentialsProvider.h"

#include <condition_variable>
#include <exception>
#include <limits>
#include <utility>

#include <aws/core/Globals.h>
#include <aws/core/auth/GeneralHTTPCredentialsProvider.h>
#include <aws/core/auth/LoginCredentialsProvider.h>
#include <aws/core/auth/ProfileCredentialsProvider.h>
#include <aws/core/auth/SSOCredentialsProvider.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/UUID.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/crt/auth/Credentials.h>
#include <aws/crt/http/HttpConnection.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/AssumeRoleRequest.h>

namespace milvus_storage {
namespace {

bool IsPresent(const Aws::String& value) { return !Aws::Utils::StringUtils::Trim(value.c_str()).empty(); }

Aws::Utils::DateTime UnlimitedExpiration() {
  return Aws::Utils::DateTime((std::chrono::system_clock::time_point::max)());
}

Aws::Client::ClientConfiguration::CredentialProviderConfiguration LegacyWebIdentityConfiguration() {
  Aws::Client::ClientConfiguration::CredentialProviderConfiguration config;
  config.profile = Aws::Auth::GetConfigProfileName();
  const auto profile = Aws::Config::GetCachedConfigProfile(config.profile);
  const auto value = [](const char* name, const Aws::String& fallback) {
    auto env = Aws::Environment::GetEnv(name);
    return env.empty() ? fallback : env;
  };
  // Match the SDK default constructor, but capture identity once: rebuilding
  // a poisoned delegate must not silently resolve a different role/profile.
  config.region = value("AWS_DEFAULT_REGION", profile.GetRegion());
  config.stsCredentialsProviderConfig.roleArn = value("AWS_ROLE_ARN", profile.GetRoleArn());
  config.stsCredentialsProviderConfig.sessionName =
      value("AWS_ROLE_SESSION_NAME", profile.GetValue("role_session_name"));
  config.stsCredentialsProviderConfig.tokenFilePath =
      value("AWS_WEB_IDENTITY_TOKEN_FILE", profile.GetValue("web_identity_token_file"));
  return config;
}

class CrtCredentialSource final : public Aws::Auth::AWSCredentialsProvider {
  public:
  CrtCredentialSource(std::shared_ptr<Aws::Crt::Auth::ICredentialsProvider> source,
                      std::chrono::milliseconds timeout,
                      bool allow_static = false)
      : source_(std::move(source)), timeout_(timeout), allow_static_(allow_static) {}

  Aws::Auth::AWSCredentials GetAWSCredentials() override {
    if (!source_ || !source_->IsValid())
      return {};
    struct State {
      std::mutex mutex;
      std::condition_variable ready;
      bool complete = false;
      Aws::Auth::AWSCredentials credentials;
    };
    auto state = std::make_shared<State>();
    const bool started = source_->GetCredentials(
        [state, allow_static = allow_static_](const std::shared_ptr<Aws::Crt::Auth::Credentials>& result, int error) {
          std::lock_guard lock(state->mutex);
          if (error == 0 && result && static_cast<bool>(*result)) {
            const auto key = result->GetAccessKeyId();
            const auto secret = result->GetSecretAccessKey();
            const auto token = result->GetSessionToken();
            const auto expiration = result->GetExpirationTimepointInSeconds();
            const bool unbounded = expiration == std::numeric_limits<uint64_t>::max();
            if (key.len && secret.len && ((!unbounded && token.len) || (allow_static && unbounded))) {
              state->credentials = Aws::Auth::AWSCredentials(
                  Aws::String(reinterpret_cast<const char*>(key.ptr), key.len),
                  Aws::String(reinterpret_cast<const char*>(secret.ptr), secret.len),
                  token.len ? Aws::String(reinterpret_cast<const char*>(token.ptr), token.len) : Aws::String{},
                  unbounded ? UnlimitedExpiration() : Aws::Utils::DateTime(expiration));
            }
          }
          state->complete = true;
          state->ready.notify_all();
        });
    if (!started)
      return {};
    std::unique_lock lock(state->mutex);
    if (!state->ready.wait_for(lock, timeout_, [&state] { return state->complete; }))
      return {};
    return state->credentials;
  }

  private:
  std::shared_ptr<Aws::Crt::Auth::ICredentialsProvider> source_;
  const std::chrono::milliseconds timeout_;
  const bool allow_static_;
};

class AssumeRoleCredentialSource final : public Aws::Auth::AWSCredentialsProvider {
  public:
  AssumeRoleCredentialSource(Aws::String role,
                             Aws::String session,
                             Aws::String external_id,
                             int duration,
                             std::shared_ptr<Aws::STS::STSClient> client)
      : client_(std::move(client)) {
    request_.WithRoleArn(std::move(role)).WithRoleSessionName(std::move(session)).WithDurationSeconds(duration);
    if (!external_id.empty())
      request_.SetExternalId(std::move(external_id));
  }
  Aws::Auth::AWSCredentials GetAWSCredentials() override {
    const auto result = client_->AssumeRole(request_);
    if (!result.IsSuccess()) {
      AWS_LOGSTREAM_WARN("MilvusAwsCredentials", "AssumeRole refresh failed: " << result.GetError().GetExceptionName());
      return {};
    }
    const auto& credentials = result.GetResult().GetCredentials();
    return {credentials.GetAccessKeyId(), credentials.GetSecretAccessKey(), credentials.GetSessionToken(),
            credentials.GetExpiration()};
  }

  private:
  std::shared_ptr<Aws::STS::STSClient> client_;
  Aws::STS::Model::AssumeRoleRequest request_;
};

class ProtectedDefaultChain final : public Aws::Auth::AWSCredentialsProviderChain {
  public:
  explicit ProtectedDefaultChain(const Aws::Client::ClientConfiguration::CredentialProviderConfiguration& config,
                                 bool configured = true) {
    // Let the SDK choose eligible sources, including container-vs-IMDS policy.
    const auto selected = configured ? std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>(config)
                                     : std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
    for (const auto& provider : selected->GetProviders()) {
      using namespace Aws::Auth;
      ValidatedAwsCredentialsProvider::Factory factory;
      bool allow_static = false;
      if (std::dynamic_pointer_cast<EnvironmentAWSCredentialsProvider>(provider)) {
        AddProvider(provider);
        continue;
      }
      if (std::dynamic_pointer_cast<InstanceProfileCredentialsProvider>(provider)) {
        AddProvider(MakeAwsInstanceProfileCredentialsProvider());
        continue;
      }
      if (std::dynamic_pointer_cast<STSAssumeRoleWebIdentityCredentialsProvider>(provider)) {
        factory = [identity = configured ? config : LegacyWebIdentityConfiguration()] {
          return std::make_shared<STSAssumeRoleWebIdentityCredentialsProvider>(identity);
        };
      } else if (std::dynamic_pointer_cast<ProfileCredentialsProvider>(provider)) {
        // Role-free profiles and credential_process explicitly support static
        // results without expiration. Once a temporary tuple was established,
        // GetAWSCredentials rejects a later loss of its expiration.
        const auto profile = Aws::Config::GetCachedConfigProfile(config.profile);
        allow_static = profile.GetRoleArn().empty();
        Aws::Crt::Auth::CredentialsProviderProfileConfig profile_config;
        const auto config_path = Aws::Auth::GetConfigProfileFilename();
        const auto credentials_path = ProfileCredentialsProvider::GetCredentialsProfileFilename();
        profile_config.ProfileNameOverride = Aws::Crt::ByteCursorFromCString(config.profile.c_str());
        profile_config.ConfigFileNameOverride = Aws::Crt::ByteCursorFromCString(config_path.c_str());
        profile_config.CredentialsFileNameOverride = Aws::Crt::ByteCursorFromCString(credentials_path.c_str());
        profile_config.Bootstrap = Aws::GetDefaultClientBootstrap();
        const auto tls = Aws::GetDefaultTlsConnectionOptions();
        if (tls)
          profile_config.TlsConnectionOptions = *tls;
        if (config.allowSystemProxy) {
          Aws::Crt::Http::ProxyEnvVarOptions proxy;
          proxy.proxyEnvVarType = Aws::Crt::Http::ProxyEnvVarType::Enabled;
          proxy.connectionType = Aws::Crt::Http::AwsHttpProxyConnectionType::Legacy;
          if (tls)
            proxy.TlsOptions = *tls;
          profile_config.ProxyEnvVarOptions = proxy;
        }
        // Use the same raw CRT provider as the SDK, with its identity/file
        // paths captured once. It has no poisoned C++ token cache to recreate.
        const auto source = Aws::Crt::Auth::CredentialsProvider::CreateCredentialsProviderProfile(profile_config);
        factory = [source, allow_static] {
          return std::make_shared<CrtCredentialSource>(source, std::chrono::minutes(5), allow_static);
        };
      } else if (std::dynamic_pointer_cast<SSOCredentialsProvider>(provider)) {
        factory = [config] { return std::make_shared<SSOCredentialsProvider>(config.profile); };
      } else if (std::dynamic_pointer_cast<LoginCredentialsProvider>(provider)) {
        factory = [config] { return std::make_shared<LoginCredentialsProvider>(config); };
      } else if (std::dynamic_pointer_cast<GeneralHTTPCredentialsProvider>(provider)) {
        const auto relative =
            Aws::Environment::GetEnv(GeneralHTTPCredentialsProvider::AWS_CONTAINER_CREDENTIALS_RELATIVE_URI);
        const auto absolute =
            Aws::Environment::GetEnv(GeneralHTTPCredentialsProvider::AWS_CONTAINER_CREDENTIALS_FULL_URI);
        const auto token = Aws::Environment::GetEnv(GeneralHTTPCredentialsProvider::AWS_CONTAINER_AUTHORIZATION_TOKEN);
        const auto token_file =
            Aws::Environment::GetEnv(GeneralHTTPCredentialsProvider::AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE);
        factory = [relative, absolute, token, token_file] {
          return std::make_shared<GeneralHTTPCredentialsProvider>(relative, absolute, token, token_file);
        };
      } else {
        // Future SDK sources still get validation; preserve SDK ordering.
        factory = [provider] { return provider; };
      }
      AddProvider(std::make_shared<ValidatedAwsCredentialsProvider>(std::move(factory), allow_static));
    }
  }
};

}  // namespace

ValidatedAwsCredentialsProvider::ValidatedAwsCredentialsProvider(Factory factory,
                                                                 bool allow_static_credentials,
                                                                 Clock clock,
                                                                 std::chrono::milliseconds refresh_grace)
    : factory_(std::move(factory)),
      allow_static_credentials_(allow_static_credentials),
      clock_(clock ? std::move(clock) : Clock([] { return Aws::Utils::DateTime::Now(); })),
      refresh_grace_(refresh_grace) {}

bool ValidatedAwsCredentialsProvider::IsUsable(const Aws::Auth::AWSCredentials& value,
                                               const Aws::Utils::DateTime& now) const {
  if (!IsPresent(value.GetAWSAccessKeyId()) || !IsPresent(value.GetAWSSecretKey()))
    return false;
  const auto expiration = value.GetExpiration();
  if (expiration == UnlimitedExpiration())
    return allow_static_credentials_;
  return IsPresent(value.GetSessionToken()) && expiration.WasParseSuccessful() && expiration > now;
}

Aws::Auth::AWSCredentials ValidatedAwsCredentialsProvider::GetAWSCredentials() {
  std::lock_guard lock(mutex_);
  const auto now = clock_();
  if (IsUsable(credentials_, now) &&
      (credentials_.GetExpiration() == UnlimitedExpiration() || credentials_.GetExpiration() - now > refresh_grace_)) {
    return credentials_;
  }
  try {
    if (!delegate_)
      delegate_ = factory_();
    const auto candidate = delegate_ ? delegate_->GetAWSCredentials() : Aws::Auth::AWSCredentials{};
    const bool lost_expiration = candidate.GetExpiration() == UnlimitedExpiration() && !credentials_.IsEmpty() &&
                                 credentials_.GetExpiration() != UnlimitedExpiration();
    if (!lost_expiration && IsUsable(candidate, clock_())) {
      credentials_ = candidate;
    } else if (!candidate.GetAWSAccessKeyId().empty() || !candidate.GetAWSSecretKey().empty() ||
               !candidate.GetSessionToken().empty()) {
      // Some SDK providers cache incomplete success results with future or
      // unlimited expiry. Recreate only that source so a later good fetch can
      // recover; never publish the invalid candidate over our usable cache.
      delegate_.reset();
    }
  } catch (const std::exception&) {
    // Preserve the validated cache on provider exceptions. Without an unexpired
    // tuple, return empty credentials under the existing AWS SDK contract.
    AWS_LOGSTREAM_WARN("MilvusAwsCredentials",
                       "Credential refresh raised an exception; retaining only unexpired cache");
  }
  return IsUsable(credentials_, clock_()) ? credentials_ : Aws::Auth::AWSCredentials{};
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsCrtCredentialsProvider(
    std::shared_ptr<Aws::Crt::Auth::ICredentialsProvider> source,
    std::chrono::milliseconds timeout,
    ValidatedAwsCredentialsProvider::Clock clock) {
  return std::make_shared<ValidatedAwsCredentialsProvider>(
      [source = std::move(source), timeout] { return std::make_shared<CrtCredentialSource>(source, timeout); }, false,
      std::move(clock));
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsWebIdentityCredentialsProvider() {
  const auto identity = LegacyWebIdentityConfiguration();
  return std::make_shared<ValidatedAwsCredentialsProvider>(
      [identity] { return std::make_shared<Aws::Auth::STSAssumeRoleWebIdentityCredentialsProvider>(identity); });
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsInstanceProfileCredentialsProvider() {
  Aws::Crt::Auth::CredentialsProviderImdsConfig config;
  config.Bootstrap = Aws::GetDefaultClientBootstrap();
  return MakeAwsCrtCredentialsProvider(Aws::Crt::Auth::CredentialsProvider::CreateCredentialsProviderImds(config));
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsDefaultCredentialsProviderChain(
    const Aws::Client::ClientConfiguration::CredentialProviderConfiguration& config) {
  return std::make_shared<ProtectedDefaultChain>(config);
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsDefaultCredentialsProviderChain() {
  Aws::Client::ClientConfiguration config(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
  return std::make_shared<ProtectedDefaultChain>(config.ResolveCredentialProviderConfig(), false);
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> MakeAwsAssumeRoleCredentialsProvider(
    const Aws::String& role_arn,
    const Aws::String& session_name,
    const Aws::String& external_id,
    int duration_seconds,
    std::shared_ptr<Aws::STS::STSClient> client,
    ValidatedAwsCredentialsProvider::Clock clock) {
  if (!client) {
    Aws::Client::ClientConfiguration config;
    client = std::make_shared<Aws::STS::STSClient>(
        MakeAwsDefaultCredentialsProviderChain(config.ResolveCredentialProviderConfig()), config);
  }
  const auto session = session_name.empty() ? Aws::String(Aws::Utils::UUID::RandomUUID()) : session_name;
  return std::make_shared<ValidatedAwsCredentialsProvider>(
      [role_arn, session, external_id, duration_seconds, client = std::move(client)] {
        return std::make_shared<AssumeRoleCredentialSource>(role_arn, session, external_id, duration_seconds, client);
      },
      false, std::move(clock), std::chrono::seconds(60));
}

}  // namespace milvus_storage
