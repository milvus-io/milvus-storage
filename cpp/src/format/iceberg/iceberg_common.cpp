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

#include <filesystem>
#include <folly/json/json.h>

namespace milvus_storage::iceberg {

// Production planning receives an already-resolved C++ filesystem. Keep these
// options credential-free: Rust uses them only to bind Iceberg URIs to that
// filesystem's local root or remote bucket/container.
std::unordered_map<std::string, std::string> ToReaderOptions(const ArrowFileSystemConfig& config) {
  std::unordered_map<std::string, std::string> options;
  const bool is_local = config.storage_type == "local";
  options["milvus_fs_is_local"] = is_local ? "true" : "false";
  if (is_local) {
    // Use one normalized absolute root for every metadata, manifest, and data
    // path check performed by the Rust URI binding.
    auto root = std::filesystem::path(config.root_path);
    if (root.is_relative()) {
      root = std::filesystem::absolute(root);
    }
    options["milvus_fs_root_path"] = root.lexically_normal().string();
  } else {
    // The C++ filesystem is already scoped to this bucket/container. Rust
    // verifies the authority before stripping it to a relative filesystem key.
    options["milvus_fs_bucket"] = config.bucket_name;
  }
  return options;
}

// Convert provider-standard Iceberg paths to the external URI form consumed by
// C++ filesystem resolution.
std::string ToMilvusUri(const std::string& standard_uri, const std::string& address) {
  if (address.empty()) {
    return standard_uri;
  }
  auto parsed = StorageUri::Parse(standard_uri, false);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return standard_uri;
  }
  // Reattach the configured filesystem address so C++ readers can resolve the
  // matching extfs entry from scheme://address/bucket/key.
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
      // Delete files are opened later by C++, so expose the same external URI
      // form used for the corresponding data file.
      entry["path"] = ToMilvusUri(path, address);
    }
  }
  return folly::toJson(parsed);
}

// This path is used only by CreateTestTable and related tests/benchmarks.
// Production planning reads through the bound C++ filesystem instead. Keep the
// native writer policy aligned with Lance: allow static AK/SK or default IAM,
// and reject provider-specific role, broker, or impersonation flows.
arrow::Result<std::unordered_map<std::string, std::string>> ToWriterOptions(const ArrowFileSystemConfig& config) {
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
    auto endpoint = StorageUri::BuildEndpointUrl(address, config.use_ssl);
    options[key] = endpoint;
    if (endpoint.find("http://") == 0)
      options["allow_http"] = "true";
  };

  // A non-IAM configuration is usable only when both credential values exist.
  const bool use_ak_sk = !config.use_iam && !config.access_key_id.empty() && !config.access_key_value.empty();
  const bool use_supported_credentials = config.use_iam || use_ak_sk;
  const auto& provider = config.cloud_provider;
  if (provider == kCloudProviderAWS) {
    // role_arn selects AssumeRole, which intentionally falls through to the
    // common NotImplemented result below.
    if (use_supported_credentials && config.role_arn.empty()) {
      if (use_ak_sk) {
        set("s3.access-key-id", config.access_key_id);
        set("s3.secret-access-key", config.access_key_value);
      }
      set("s3.region", config.region);
      set_endpoint("s3.endpoint", config.address);
      return options;
    }
  }
  if (provider == kCloudProviderAzure) {
    // Broker-issued SAS credentials belong to the C++ filesystem read path and
    // are not reconstructed by the native writer.
    if (use_supported_credentials && !config.IsAzureCredentialBrokerEnabled()) {
      set("adls.account-name", config.access_key_id);
      set("adls.endpoint-suffix", config.address);
      if (use_ak_sk) {
        set("adls.account-key", config.access_key_value);
      }
      return options;
    }
  }
  if (provider == kCloudProviderGCP) {
    // A target service account requests impersonation; only default IAM is
    // supported by this native writer path.
    if (config.use_iam && config.gcp_target_service_account.empty()) {
      return options;
    }
  }
  if (provider == kCloudProviderAliyun) {
    // role_arn selects an Aliyun role flow and is rejected like AWS AssumeRole.
    if (use_supported_credentials && config.role_arn.empty()) {
      if (use_ak_sk) {
        set("oss.access-key-id", config.access_key_id);
        set("oss.access-key-secret", config.access_key_value);
      }
      set("oss.region", config.region);
      set_endpoint("oss.endpoint", config.address);
      return options;
    }
  }
  return arrow::Status::NotImplemented("Unsupported Iceberg native writer configuration for cloud provider: ",
                                       provider);
}

// Normalize provider and Milvus URI forms before comparing Iceberg data and
// delete file paths.
std::string StripAbfssEndpoint(const std::string& uri) {
  const auto scheme_end = uri.find("://");
  if (scheme_end == std::string::npos) {
    return uri;
  }
  const auto scheme = uri.substr(0, scheme_end);
  // Iceberg metadata may use any Azure alias supported by the bound
  // filesystem adapter, including legacy WASB forms.
  if (scheme != "azure" && scheme != "abfs" && scheme != "abfss" && scheme != "wasb" && scheme != "wasbs") {
    return uri;
  }
  const auto authority_start = scheme_end + 3;
  // Only look for '@' in the authority (before the first '/'), not in the path.
  // Paths can legitimately contain '@' (e.g. wasbs://container/user@org/file).
  const auto first_slash = uri.find('/', authority_start);
  const auto authority_end = (first_slash == std::string::npos) ? uri.size() : first_slash;
  const auto at_pos = uri.find('@', authority_start);
  if (at_pos == std::string::npos || at_pos >= authority_end) {
    return uri;  // no @ in authority
  }
  // Preserve the recorded alias while removing its account endpoint.
  return uri.substr(0, authority_start) + uri.substr(authority_start, at_pos - authority_start) +
         uri.substr(authority_end);
}

std::string MilvusURIToIcebergURI(const std::string& uri) {
  // Two mutually exclusive cases:
  // 1. Azure recorded format: scheme://container@endpoint/path → strip @endpoint
  // 2. Milvus format:         scheme://address/bucket/path    → strip address
  // They cannot be chained: after stripping @endpoint the result is
  // scheme://container/path where "container" is NOT an address.
  const auto stripped = StripAbfssEndpoint(uri);
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
