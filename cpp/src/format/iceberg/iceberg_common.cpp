// Copyright 2025 Zilliz
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

#include "milvus-storage/format/iceberg/iceberg_common.h"

#include <cstdlib>
#include <fmt/format.h>
#include <folly/json/json.h>
#include "milvus-storage/common/log.h"

namespace milvus_storage::iceberg {

std::string ToMilvusUri(const std::string& standard_uri, const std::string& address) {
  if (address.empty()) {
    return standard_uri;
  }
  auto parsed = StorageUri::Parse(standard_uri, false);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return standard_uri;
  }
  parsed->address = address;
  auto result = StorageUri::Make(parsed.ValueOrDie());
  return result.ok() ? result.ValueOrDie() : standard_uri;
}

std::string ConvertDeleteMetadataPaths(const std::vector<uint8_t>& json_bytes, const std::string& address) {
  if (address.empty()) {
    return std::string(json_bytes.begin(), json_bytes.end());
  }
  std::string json_str(json_bytes.begin(), json_bytes.end());
  auto parsed = folly::parseJson(json_str);
  for (auto& entry : parsed) {
    auto path = entry.getDefault("path", "").asString();
    if (!path.empty()) {
      entry["path"] = ToMilvusUri(path, address);
    }
  }
  return folly::toJson(parsed);
}

std::unordered_map<std::string, std::string> ToStorageOptions(const ArrowFileSystemConfig& config) {
  std::unordered_map<std::string, std::string> options;
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
      // AssumeRole: set ARN fields + region/endpoint; do NOT set AKSK so opendal
      // uses the default credential chain (EC2 metadata / env vars) as base
      // credential for the STS AssumeRole call.
      set("s3.region", config.region);
      set_endpoint("s3.endpoint", config.address);
      set("client.assume-role.arn", config.role_arn);
      set("client.assume-role.session-name", config.session_name);
      set("client.assume-role.external-id", config.external_id);
    } else {
      // Explicit AKSK or IAM
      if (!config.use_iam) {
        set("s3.access-key-id", config.access_key_id);
        set("s3.secret-access-key", config.access_key_value);
      }
      set("s3.region", config.region);
      set_endpoint("s3.endpoint", config.address);
    }
  } else if (provider == kCloudProviderAzure) {
    set("adls.account-name", config.access_key_id);
    // Pass the endpoint suffix so the Rust bridge can reconstruct the full
    // Azure DFS endpoint (account.dfs.suffix) from scheme://container/path URIs.
    set("adls.endpoint-suffix", config.address);
    if (config.use_iam) {
      auto* client_id = std::getenv("AZURE_CLIENT_ID");
      if (client_id)
        set("adls.client-id", client_id);
      auto* tenant_id = std::getenv("AZURE_TENANT_ID");
      if (tenant_id)
        set("adls.tenant-id", tenant_id);
    } else {
      set("adls.account-key", config.access_key_value);
    }
  } else if (provider == kCloudProviderGCP) {
    if (!config.gcp_target_service_account.empty()) {
      // Bridge-private: iceberg-rust 0.8's gcs_config_parse doesn't recognize
      // this key as an impersonation target (it would be silently dropped).
      // Instead, iceberg_bridgeimpl.rs::iceberg_plan_files intercepts it,
      // fetches a token via VM-SA → IAM.generateAccessToken, and swaps it for
      // `gcs.oauth2.token` before building the FileIO. See
      // `docs/iceberg-gcp-impersonation-analysis.md`.
      set("gcs.service-account", config.gcp_target_service_account);
    }
    // Otherwise uses default credentials (VM metadata)
  } else if (provider == kCloudProviderAliyun) {
    if (!config.role_arn.empty()) {
      // Per-tenant AssumeRoleWithOIDC. Machine identity (oidc_token_file,
      // oidc_provider_arn) stays in process env — opendal picks it up via the
      // env sweep inside `AliyunOssStorage::create_operator`. Do NOT emit
      // `oss.access-key-id` / `oss.access-key-secret` on this branch:
      // reqsign's `load_via_static` runs before `load_via_assume_role_with_oidc`,
      // so static creds would silently bypass the OIDC path. Mirrors the
      // Aliyun role_arn branch in `lance_common.cpp`.
      //
      // `oss.role-arn` / `oss.role-session-name` are bridge-private keys
      // consumed only by `AliyunOssStorage` on the Rust side — stock
      // iceberg-storage-opendal's `OssConfig` would drop them, which is the
      // whole reason we route `oss://` through our own `StorageFactory`
      // instead of `OpenDalStorageFactory::Oss`.
      set_endpoint("oss.endpoint", config.address);
      set("oss.region", config.region);
      set("oss.role-arn", config.role_arn);
      set("oss.role-session-name", config.session_name);
    } else {
      // Explicit AKSK. iceberg 0.9 `OSS_*` constants in the `iceberg` crate
      // map to these three dotted keys.
      set("oss.access-key-id", config.access_key_id);
      set("oss.access-key-secret", config.access_key_value);
      set("oss.region", config.region);
      set_endpoint("oss.endpoint", config.address);
    }
  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    throw std::runtime_error("Unsupported cloud provider: " + provider);
  } else {
    throw std::runtime_error("Unknown cloud provider: " + provider);
  }
  return options;
}

std::string StripAbfssEndpoint(const std::string& uri) {
  // Only process abfss:// or abfs:// URIs
  auto scheme_end = uri.find("://");
  if (scheme_end == std::string::npos) {
    return uri;
  }
  auto scheme = uri.substr(0, scheme_end);
  if (scheme != "abfss" && scheme != "abfs") {
    return uri;
  }
  auto authority_start = scheme_end + 3;
  // Only look for '@' in the authority (before the first '/'), not in the path.
  // Paths can legitimately contain '@' (e.g. abfss://container/user@org/file).
  auto first_slash = uri.find('/', authority_start);
  auto authority_end = (first_slash == std::string::npos) ? uri.size() : first_slash;
  auto at_pos = uri.find('@', authority_start);
  if (at_pos == std::string::npos || at_pos >= authority_end) {
    return uri;  // no @ in authority
  }
  // abfss://container@endpoint/path → abfss://container/path
  return uri.substr(0, authority_start) + uri.substr(authority_start, at_pos - authority_start) +
         uri.substr(authority_end);
}

std::string MilvusURIToIcebergURI(const std::string& uri) {
  // Two mutually exclusive cases:
  // 1. ABFSS opendal format: abfss://container@endpoint/path → strip @endpoint
  // 2. Milvus format:        scheme://address/bucket/path    → strip address
  // They cannot be chained: after stripping @endpoint the result is
  // abfss://container/path where "container" is NOT an address.
  auto stripped = StripAbfssEndpoint(uri);
  if (stripped != uri) {
    return stripped;  // case 1: had @, stripped it, done
  }
  // case 2: no @ found, try stripping Milvus address
  auto parsed = StorageUri::Parse(uri);
  if (parsed.ok() && !parsed->scheme.empty() && !parsed->address.empty()) {
    auto result = StorageUri::Make(parsed.ValueOrDie(), false);
    if (result.ok()) {
      return result.ValueOrDie();
    }
  }
  return uri;
}

}  // namespace milvus_storage::iceberg
