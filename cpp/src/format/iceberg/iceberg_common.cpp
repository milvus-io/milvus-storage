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

#include <folly/json/json.h>

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
    if (!value.empty()) options[key] = value;
  };
  auto set_endpoint = [&](const std::string& key, const std::string& address) {
    if (address.empty()) return;
    options[key] = StorageUri::BuildEndpointUrl(address);
    if (address.find("http://") == 0) options["allow_http"] = "true";
  };

  const auto& provider = config.cloud_provider;
  if (provider == kCloudProviderAWS) {
    set("s3.access-key-id", config.access_key_id);
    set("s3.secret-access-key", config.access_key_value);
    set("s3.region", config.region);
    set_endpoint("s3.endpoint", config.address);
  } else if (provider == kCloudProviderAzure) {
    set("adls.account-name", config.access_key_id);
    set("adls.account-key", config.access_key_value);
  } else if (provider == kCloudProviderGCP) {
    // GCP uses default credentials
  } else if (provider == kCloudProviderAliyun) {
    set("oss.access-key-id", config.access_key_id);
    set("oss.access-key-secret", config.access_key_value);
    set_endpoint("oss.endpoint", config.address);
  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    throw std::runtime_error("Unsupported cloud provider: " + provider);
  } else {
    throw std::runtime_error("Unknown cloud provider: " + provider);
  }
  return options;
}

}  // namespace milvus_storage::iceberg
