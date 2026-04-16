// Copyright 2023 Zilliz
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

#include "milvus-storage/format/lance/lance_common.h"

#include <cstdlib>
#include <fmt/format.h>
#include "milvus-storage/common/log.h"

namespace milvus_storage::lance {

//------------------------------------------------------------------------------
// Storage Options
//------------------------------------------------------------------------------

StorageOptions ToStorageOptions(const ArrowFileSystemConfig& config) {
  StorageOptions options;
  if (config.storage_type == "local") {
    return options;
  }

  auto set = [&](const std::string& key, const std::string& value) {
    if (!value.empty())
      options[key] = value;
  };
  auto set_endpoint = [&](const std::string& key, const std::string& address) {
    if (address.empty())
      return;
    options[key] = StorageUri::BuildEndpointUrl(address);
    if (address.find("http://") == 0)
      options["allow_http"] = "true";
  };

  const auto& provider = config.cloud_provider;
  LOG_STORAGE_DEBUG_ << fmt::format(
      "provider={}, endpoint={}, use_ssl={}, use_iam={}, has_aksk={}, role_arn={}, gcp_target_sa={}", provider,
      config.address, config.use_ssl, config.use_iam, !config.access_key_id.empty() && !config.access_key_value.empty(),
      config.role_arn.empty() ? "(empty)" : config.role_arn,
      config.gcp_target_service_account.empty() ? "(empty)" : config.gcp_target_service_account);
  if (provider == kCloudProviderAWS) {
    if (!config.role_arn.empty()) {
      // AssumeRole: set region/endpoint + ARN fields; do NOT set AKSK so the
      // Rust layer uses the default credential chain (EC2 metadata / env vars)
      // as base credential for the STS AssumeRole call.
      set("aws_region", config.region);
      set_endpoint("aws_endpoint", config.address);
      set("aws_role_arn", config.role_arn);
      set("aws_session_name", config.session_name);
      set("aws_external_id", config.external_id);
      if (config.load_frequency > 0) {
        options["aws_credential_refresh_secs"] = std::to_string(config.load_frequency);
      }
    } else {
      // Explicit AKSK or IAM
      if (!config.use_iam) {
        set("aws_access_key_id", config.access_key_id);
        set("aws_secret_access_key", config.access_key_value);
      }
      set("aws_region", config.region);
      set_endpoint("aws_endpoint", config.address);
    }
  } else if (provider == kCloudProviderAzure) {
    set("azure_storage_account_name", config.access_key_id);
    if (!config.use_iam) {
      set("azure_storage_account_key", config.access_key_value);
    }
    if (!config.address.empty()) {
      const char* azurite_env = std::getenv("USE_AZURITE");
      std::string blob_authority =
          (azurite_env && std::string(azurite_env) == "true") ? config.address : ".blob." + config.address;
      options["azure_endpoint"] =
          StorageUri::BuildAzureEndpointAddress(blob_authority, config.access_key_id, config.use_ssl);
      if (!config.use_ssl)
        options["allow_http"] = "true";
    }
  } else if (provider == kCloudProviderGCP) {
    if (!config.gcp_target_service_account.empty()) {
      // Bridge-private keys consumed by Rust `open_dataset`/`write_dataset`/
      // `drop` (see lance_bridgeimpl.rs). The bridge strips them out of
      // storage_options and installs an ImpersonatingGcsStoreProvider that
      // hands lance-io a CredentialProvider doing
      //   VM default SA token (metadata.google.internal)
      //     → IAM Credentials generateAccessToken(target_sa)
      //     → GcpCredential { bearer: <impersonated token> }
      // with token caching and refresh ahead of expiry. Neither object_store
      // (lance default) nor opendal natively supports VM-SA→target-SA
      // impersonation via a config key, hence the custom provider.
      set("gcp_target_service_account", config.gcp_target_service_account);
      if (config.load_frequency > 0) {
        // TTL requested from generateAccessToken; the credential provider
        // refreshes well before this elapses. Mirrors aws_credential_refresh_secs.
        options["gcp_credential_refresh_secs"] = std::to_string(config.load_frequency);
      }
    }
    // Otherwise uses default credentials (VM metadata)
  } else if (provider == kCloudProviderAliyun) {
    set("oss_access_key_id", config.access_key_id);
    set("oss_secret_access_key", config.access_key_value);
    set("oss_region", config.region);
    set_endpoint("oss_endpoint", config.address);
  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    throw std::runtime_error("Unsupported cloud provider: " + provider);
  } else {
    throw std::runtime_error("Unknown cloud provider: " + provider);
  }
  return options;
}

//------------------------------------------------------------------------------
// URI Parsing and Construction
//------------------------------------------------------------------------------

static const std::string kLanceUriDelimiter = "?fragment_id=";

arrow::Result<std::pair<std::string, uint64_t>> ParseLanceUri(const std::string& uri) {
  auto pos = uri.find(kLanceUriDelimiter);
  if (pos == std::string::npos) {
    return arrow::Status::Invalid("Invalid uri format: ", uri,
                                  ". Expected format: {base_path}?fragment_id={fragment_id}");
  }

  uint64_t fragment_id = 0;
  try {
    fragment_id = std::stoull(uri.substr(pos + kLanceUriDelimiter.length()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Invalid fragment_id in uri: {}", uri));
  }

  auto base_path = uri.substr(0, pos);
  return std::make_pair(base_path, fragment_id);
}

std::string MakeLanceUri(const std::string& base_path, uint64_t fragment_id) {
  return base_path + kLanceUriDelimiter + std::to_string(fragment_id);
}

//------------------------------------------------------------------------------
// Cloud Provider URI Scheme Mapping
//------------------------------------------------------------------------------

static arrow::Result<std::string> GetCloudUriScheme(const std::string& provider) {
  if (provider == kCloudProviderAWS) {
    return "s3";
  }
  if (provider == kCloudProviderAzure) {
    return "az";
  }
  if (provider == kCloudProviderGCP) {
    return "gs";
  }
  if (provider == kCloudProviderAliyun) {
    return "oss";
  }
  if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    return arrow::Status::Invalid("Lance does not support cloud provider: " + provider);
  }
  return arrow::Status::Invalid("Unknown cloud provider: " + provider);
}

arrow::Result<std::string> BuildLanceBaseUri(const ArrowFileSystemConfig& config, const std::string& relative_path) {
  if (config.storage_type == "local") {
    return config.root_path + "/" + relative_path;
  }

  if (config.bucket_name.empty()) {
    return arrow::Status::Invalid("Bucket name is required for cloud storage");
  }

  ARROW_ASSIGN_OR_RAISE(auto scheme, GetCloudUriScheme(config.cloud_provider));
  return scheme + "://" + config.bucket_name + "/" + relative_path;
}

std::string ToMilvusLanceUri(const std::string& standard_uri, const std::string& address) {
  if (address.empty()) {
    return standard_uri;
  }
  auto parsed = StorageUri::Parse(standard_uri, /*include_address=*/false);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return standard_uri;
  }
  parsed->address = address;
  auto result = StorageUri::Make(parsed.ValueOrDie(), /*include_address=*/true);
  return result.ok() ? result.ValueOrDie() : standard_uri;
}

std::string ToStandardLanceUri(const std::string& milvus_uri) {
  auto parsed = StorageUri::Parse(milvus_uri, /*include_address=*/true);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return milvus_uri;
  }
  auto result = StorageUri::Make(parsed.ValueOrDie(), /*include_address=*/false);
  return result.ok() ? result.ValueOrDie() : milvus_uri;
}

}  // namespace milvus_storage::lance
