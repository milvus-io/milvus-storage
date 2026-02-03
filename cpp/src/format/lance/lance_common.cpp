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

#include <fmt/format.h>

namespace milvus_storage::lance {

static const std::string kLanceUriDelimiter = "?fragment_id=";

//------------------------------------------------------------------------------
// URI Parsing and Construction
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// Endpoint URL Helpers
//------------------------------------------------------------------------------

struct EndpointInfo {
  std::string url;
  bool allow_http = false;
};

static EndpointInfo BuildEndpointUrl(const std::string& address) {
  if (address.empty()) {
    return {};
  }

  // If already has scheme, check if it's http
  if (address.find("://") != std::string::npos) {
    bool is_http = (address.find("http://") == 0);
    return {address, is_http};
  }

  // Default to HTTPS for cloud storage
  return {"https://" + address, false};
}

static std::string BuildAzureEndpointAddress(const std::string& address, const std::string& account_name) {
  std::string host = address;
  std::string scheme_prefix;

  // Extract scheme if present
  size_t scheme_pos = host.find("://");
  if (scheme_pos != std::string::npos) {
    scheme_prefix = host.substr(0, scheme_pos + 3);
    host = host.substr(scheme_pos + 3);
  }

  // Prepend account name if not already present
  // Azure endpoint format: https://<account>.blob.core.windows.net
  if (!account_name.empty() && host.find(account_name + ".") != 0) {
    host = account_name + "." + host;
  }

  return scheme_prefix + host;
}

//------------------------------------------------------------------------------
// Provider-Specific Storage Options
//------------------------------------------------------------------------------

static void SetOptionIfNotEmpty(LanceStorageOptions& options, const std::string& key, const std::string& value) {
  if (!value.empty()) {
    options[key] = value;
  }
}

static void SetEndpointOptions(LanceStorageOptions& options,
                               const std::string& endpoint_key,
                               const std::string& address) {
  if (address.empty()) {
    return;
  }

  auto endpoint_info = BuildEndpointUrl(address);
  options[endpoint_key] = endpoint_info.url;
  if (endpoint_info.allow_http) {
    options["allow_http"] = "true";
  }
}

static void ConfigureAwsOptions(LanceStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "aws_access_key_id", config.access_key_id);
  SetOptionIfNotEmpty(options, "aws_secret_access_key", config.access_key_value);
  SetOptionIfNotEmpty(options, "aws_region", config.region);
  SetEndpointOptions(options, "aws_endpoint", config.address);
}

static void ConfigureAzureOptions(LanceStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "azure_storage_account_name", config.access_key_id);
  SetOptionIfNotEmpty(options, "azure_storage_account_key", config.access_key_value);

  if (!config.address.empty()) {
    auto azure_address = BuildAzureEndpointAddress(config.address, config.access_key_id);
    SetEndpointOptions(options, "azure_endpoint", azure_address);
  }
}

static void ConfigureAliyunOptions(LanceStorageOptions& options, const ArrowFileSystemConfig& config) {
  SetOptionIfNotEmpty(options, "oss_access_key_id", config.access_key_id);
  SetOptionIfNotEmpty(options, "oss_secret_access_key", config.access_key_value);
  SetOptionIfNotEmpty(options, "oss_region", config.region);
  SetEndpointOptions(options, "oss_endpoint", config.address);
}

//------------------------------------------------------------------------------
// Main Conversion Function
//------------------------------------------------------------------------------

LanceStorageOptions ToLanceStorageOptions(const ArrowFileSystemConfig& config) {
  LanceStorageOptions options;

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
    throw LanceException("Lance does not support cloud provider: " + provider);
  } else {
    throw LanceException("Unknown cloud provider: " + provider);
  }

  return options;
}

}  // namespace milvus_storage::lance
