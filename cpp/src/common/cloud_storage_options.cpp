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

#include "milvus-storage/common/cloud_storage_options.h"

namespace milvus_storage {

namespace {

struct EndpointInfo {
  std::string url;
  bool allow_http = false;
};

EndpointInfo BuildEndpointUrl(const std::string& address) {
  if (address.empty()) {
    return {};
  }
  if (address.find("://") != std::string::npos) {
    bool is_http = (address.find("http://") == 0);
    return {address, is_http};
  }
  return {"https://" + address, false};
}

std::string BuildAzureEndpointAddress(const std::string& address, const std::string& account_name) {
  std::string host = address;
  std::string scheme_prefix;

  size_t scheme_pos = host.find("://");
  if (scheme_pos != std::string::npos) {
    scheme_prefix = host.substr(0, scheme_pos + 3);
    host = host.substr(scheme_pos + 3);
  }

  if (!account_name.empty() && host.find(account_name + ".") != 0) {
    host = account_name + "." + host;
  }

  return scheme_prefix + host;
}

void SetOptionIfNotEmpty(CloudStorageOptions& options, const std::string& key, const std::string& value) {
  if (!value.empty()) {
    options[key] = value;
  }
}

void SetEndpointOptions(CloudStorageOptions& options, const std::string& endpoint_key, const std::string& address) {
  if (address.empty()) {
    return;
  }
  auto endpoint_info = BuildEndpointUrl(address);
  options[endpoint_key] = endpoint_info.url;
  if (endpoint_info.allow_http) {
    options["allow_http"] = "true";
  }
}

void ConfigureAwsOptions(CloudStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "aws_access_key_id", config.access_key_id);
  SetOptionIfNotEmpty(options, "aws_secret_access_key", config.access_key_value);
  SetOptionIfNotEmpty(options, "aws_region", config.region);
  SetEndpointOptions(options, "aws_endpoint", config.address);
}

void ConfigureAzureOptions(CloudStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "azure_storage_account_name", config.access_key_id);
  SetOptionIfNotEmpty(options, "azure_storage_account_key", config.access_key_value);

  if (!config.address.empty()) {
    auto azure_address = BuildAzureEndpointAddress(config.address, config.access_key_id);
    SetEndpointOptions(options, "azure_endpoint", azure_address);
  }
}

void ConfigureAliyunOptions(CloudStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "oss_access_key_id", config.access_key_id);
  SetOptionIfNotEmpty(options, "oss_secret_access_key", config.access_key_value);
  SetOptionIfNotEmpty(options, "oss_region", config.region);
  SetEndpointOptions(options, "oss_endpoint", config.address);
}

}  // namespace

CloudStorageOptions ToCloudStorageOptions(const ArrowFileSystemConfig& config) {
  CloudStorageOptions options;

  if (config.storage_type == "local") {
    return options;
  }

  const auto& provider = config.cloud_provider;

  if (provider == kCloudProviderAWS) {
    ConfigureAwsOptions(options, config);
  } else if (provider == kCloudProviderAzure) {
    ConfigureAzureOptions(options, config);
  } else if (provider == kCloudProviderGCP) {
    // GCP uses default credentials, no additional options needed
  } else if (provider == kCloudProviderAliyun) {
    ConfigureAliyunOptions(options, config);
  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    throw std::runtime_error("Unsupported cloud provider: " + provider);
  } else {
    throw std::runtime_error("Unknown cloud provider: " + provider);
  }

  return options;
}

}  // namespace milvus_storage
