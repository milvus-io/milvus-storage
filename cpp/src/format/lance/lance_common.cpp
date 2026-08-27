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
#include <filesystem>

namespace milvus_storage::lance {

//------------------------------------------------------------------------------
// Storage Options
//------------------------------------------------------------------------------

StorageOptions ToReaderOptions(const ArrowFileSystemConfig& config) {
  StorageOptions options;
  options["milvus_fs_cache_key"] = config.GetCacheKey();
  options["milvus_fs_is_local"] = config.storage_type == "local" ? "true" : "false";
  if (config.storage_type == "local") {
    auto root_path = std::filesystem::path(config.root_path);
    if (root_path.is_relative()) {
      root_path = std::filesystem::absolute(root_path);
    }
    options["milvus_fs_root_path"] = root_path.lexically_normal().string();
  }
  options["lance_io_parallelism"] = std::to_string(config.lance_io_parallelism);
  options["lance_aimd_initial_rate"] = std::to_string(config.iops_initial_rate);
  options["lance_aimd_max_rate"] = std::to_string(config.iops_max_rate);
  return options;
}

arrow::Result<StorageOptions> ToWriterOptions(const ArrowFileSystemConfig& config) {
  StorageOptions options;
  if (config.storage_type == "local") {
    return options;
  }
  options["lance_aimd_initial_rate"] = std::to_string(config.iops_initial_rate);
  options["lance_aimd_max_rate"] = std::to_string(config.iops_max_rate);

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

  const bool use_ak_sk = !config.use_iam && !config.access_key_id.empty() && !config.access_key_value.empty();
  const bool use_supported_credentials = config.use_iam || use_ak_sk;
  const auto& provider = config.cloud_provider;
  if (provider == kCloudProviderAWS) {
    if (use_supported_credentials && config.role_arn.empty()) {
      if (use_ak_sk) {
        set("aws_access_key_id", config.access_key_id);
        set("aws_secret_access_key", config.access_key_value);
      }
      set("aws_region", config.region);
      set_endpoint("aws_endpoint", config.address);
      return options;
    }
  }
  if (provider == kCloudProviderAzure) {
    if (use_supported_credentials && !config.IsAzureCredentialBrokerEnabled()) {
      set("azure_storage_account_name", config.access_key_id);
      if (use_ak_sk) {
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
      return options;
    }
  }
  if (provider == kCloudProviderGCP) {
    if (config.use_iam && config.gcp_target_service_account.empty()) {
      return options;
    }
  }
  if (provider == kCloudProviderAliyun) {
    if (use_supported_credentials && config.role_arn.empty()) {
      if (use_ak_sk) {
        set("oss_access_key_id", config.access_key_id);
        set("oss_secret_access_key", config.access_key_value);
      }
      set("oss_region", config.region);
      set_endpoint("oss_endpoint", config.address);
      return options;
    }
  }
  return arrow::Status::NotImplemented("Unsupported Lance native writer configuration for cloud provider: ", provider);
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
