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

#include "milvus-storage/filesystem/fs.h"

#include <memory>
#include <mutex>
#include <stdexcept>

#include <arrow/filesystem/localfs.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/common/path_util.h"
#include "milvus-storage/common/lrucache.h"

#ifdef MILVUS_AZURE_FS
#include "milvus-storage/filesystem/azure/azure_fs.h"
#endif

namespace milvus_storage {

static constexpr auto local_uri_scheme = "file://";

arrow::Result<ArrowFileSystemPtr> CreateArrowFileSystem(const ArrowFileSystemConfig& config) {
  std::string out_path;
  auto storage_type = StorageType_Map[config.storage_type];
  switch (storage_type) {
    case StorageType::Local: {
      auto path = boost::filesystem::path(config.root_path);
      if (path.is_relative()) {
        path = boost::filesystem::absolute(path);
      }
      std::string local_uri = local_uri_scheme + path.string();

      ARROW_ASSIGN_OR_RAISE(auto arrow_uri, arrow::util::Uri::FromString(local_uri));
      ARROW_ASSIGN_OR_RAISE(auto option, arrow::fs::LocalFileSystemOptions::FromUri(arrow_uri, &out_path));

      // create local dir if not exists
      // if exists, check it is a directory
      boost::filesystem::path dir_path(out_path);
      if (!boost::filesystem::exists(dir_path)) {
        boost::filesystem::create_directories(dir_path);
      } else if (!boost::filesystem::is_directory(dir_path)) {
        return arrow::Status::Invalid("Path ", out_path, " is not a directory");
      }

      return std::make_shared<arrow::fs::SubTreeFileSystem>(out_path,
                                                            std::make_shared<arrow::fs::LocalFileSystem>(option));
    }
    case StorageType::Remote: {
      auto cloud_provider = CloudProviderType_Map[config.cloud_provider];
      switch (cloud_provider) {
#ifdef MILVUS_AZURE_FS
        case CloudProviderType::AZURE: {
          auto producer = std::make_shared<AzureFileSystemProducer>(config);
          return producer->Make();
        }
#endif
        case CloudProviderType::AWS:
        case CloudProviderType::GCP:
        case CloudProviderType::ALIYUN:
        case CloudProviderType::TENCENTCLOUD:
        case CloudProviderType::HUAWEICLOUD: {
          auto producer = std::make_shared<S3FileSystemProducer>(config);
          return producer->Make();
        }
        default: {
          return arrow::Status::Invalid("Unsupported cloud provider: " + config.cloud_provider);
        }
      }
    }
    default: {
      return arrow::Status::Invalid("Unsupported storage type: " + config.storage_type);
    }
  }
}

// ==================== FilesystemCache Implementation ====================

FilesystemCache& FilesystemCache::getInstance() {
  static FilesystemCache instance(16);  // Default capacity
  return instance;
}

size_t FilesystemCache::size() const { return cache_.size(); }

void FilesystemCache::remove(const std::string& key) { cache_.remove(key); }

void FilesystemCache::clean() { cache_.clean(); }

void FilesystemCache::set_capacity(size_t capacity) { cache_.set_capacity(capacity); }

arrow::Status ArrowFileSystemConfig::create_file_system_config(const milvus_storage::api::Properties& properties_map,
                                                               ArrowFileSystemConfig& result) {
  ARROW_ASSIGN_OR_RAISE(result.address, api::GetValue<std::string>(properties_map, PROPERTY_FS_ADDRESS));
  ARROW_ASSIGN_OR_RAISE(result.bucket_name, api::GetValue<std::string>(properties_map, PROPERTY_FS_BUCKET_NAME));
  ARROW_ASSIGN_OR_RAISE(result.access_key_id, api::GetValue<std::string>(properties_map, PROPERTY_FS_ACCESS_KEY_ID));
  ARROW_ASSIGN_OR_RAISE(result.access_key_value,
                        api::GetValue<std::string>(properties_map, PROPERTY_FS_ACCESS_KEY_VALUE));
  ARROW_ASSIGN_OR_RAISE(result.root_path, api::GetValue<std::string>(properties_map, PROPERTY_FS_ROOT_PATH));
  ARROW_ASSIGN_OR_RAISE(result.storage_type, api::GetValue<std::string>(properties_map, PROPERTY_FS_STORAGE_TYPE));
  ARROW_ASSIGN_OR_RAISE(result.cloud_provider, api::GetValue<std::string>(properties_map, PROPERTY_FS_CLOUD_PROVIDER));
  ARROW_ASSIGN_OR_RAISE(result.iam_endpoint, api::GetValue<std::string>(properties_map, PROPERTY_FS_IAM_ENDPOINT));
  ARROW_ASSIGN_OR_RAISE(result.log_level, api::GetValue<std::string>(properties_map, PROPERTY_FS_LOG_LEVEL));
  ARROW_ASSIGN_OR_RAISE(result.region, api::GetValue<std::string>(properties_map, PROPERTY_FS_REGION));
  ARROW_ASSIGN_OR_RAISE(result.use_ssl, api::GetValue<bool>(properties_map, PROPERTY_FS_USE_SSL));
  ARROW_ASSIGN_OR_RAISE(result.ssl_ca_cert, api::GetValue<std::string>(properties_map, PROPERTY_FS_SSL_CA_CERT));
  ARROW_ASSIGN_OR_RAISE(result.use_iam, api::GetValue<bool>(properties_map, PROPERTY_FS_USE_IAM));
  ARROW_ASSIGN_OR_RAISE(result.use_virtual_host, api::GetValue<bool>(properties_map, PROPERTY_FS_USE_VIRTUAL_HOST));
  ARROW_ASSIGN_OR_RAISE(result.request_timeout_ms,
                        api::GetValue<int64_t>(properties_map, PROPERTY_FS_REQUEST_TIMEOUT_MS));
  ARROW_ASSIGN_OR_RAISE(result.gcp_native_without_auth,
                        api::GetValue<bool>(properties_map, PROPERTY_FS_GCP_NATIVE_WITHOUT_AUTH));
  ARROW_ASSIGN_OR_RAISE(result.gcp_credential_json,
                        api::GetValue<std::string>(properties_map, PROPERTY_FS_GCP_CREDENTIAL_JSON));
  ARROW_ASSIGN_OR_RAISE(result.use_custom_part_upload,
                        api::GetValue<bool>(properties_map, PROPERTY_FS_USE_CUSTOM_PART_UPLOAD));
  ARROW_ASSIGN_OR_RAISE(result.max_connections, api::GetValue<uint32_t>(properties_map, PROPERTY_FS_MAX_CONNECTIONS));
  return arrow::Status::OK();
}

// ==================== External Filesystem Support Implementation ====================

namespace {

/**
 * @brief Extract external filesystem properties from the properties map
 *
 * Parses properties matching the pattern: extfs.<name>.<property>
 * and groups them by <name>.
 *
 * @param properties Input properties map
 * @return Map of external_fs_name -> Properties with fs.* keys
 */
arrow::Result<std::unordered_map<std::string, api::Properties>> ExtractExternalFsProperties(
    const api::Properties& properties) {
  std::unordered_map<std::string, api::Properties> external_fs_map;

  const std::string prefix = PROPERTY_EXTFS_PREFIX;
  for (const auto& [key, value] : properties) {
    if (key.size() <= prefix.size() || key.substr(0, prefix.size()) != prefix) {
      continue;  // Not an external fs property
    }

    // Parse: extfs.<name>.<property>
    std::string remainder = key.substr(prefix.size());
    size_t dot_pos = remainder.find('.');
    if (dot_pos == std::string::npos) {
      return arrow::Status::Invalid("Invalid external filesystem property format: '", key,
                                    "'. Expected format: extfs.<name>.<property>");
    }

    std::string fs_name = remainder.substr(0, dot_pos);
    std::string fs_property = remainder.substr(dot_pos + 1);

    if (fs_name.empty()) {
      return arrow::Status::Invalid("Empty external filesystem name in property: '", key, "'");
    }

    if (fs_property.empty()) {
      return arrow::Status::Invalid("Empty property name in external filesystem property: '", key, "'");
    }

    // Map to standard fs.* property name
    std::string standard_key = std::string(PROPERTY_FS_PREFIX) + fs_property;
    external_fs_map[fs_name][standard_key] = value;
  }

  return external_fs_map;
}

/**
 * @brief Create an ArrowFileSystemConfig from extracted properties with alias
 */
arrow::Status CreateExternalFsConfig(const std::string& alias,
                                     const api::Properties& props,
                                     ArrowFileSystemConfig& result) {
  auto status = ArrowFileSystemConfig::create_file_system_config(props, result);
  if (status.ok()) {
    result.alias = alias;
  }
  return status;
}

}  // namespace

arrow::Result<ArrowFileSystemPtr> FilesystemCache::get(const api::Properties& properties, const std::string& path) {
  std::string cache_key;

  // If path is provided with a scheme, try to resolve external filesystem
  if (!path.empty()) {
    ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(path));

    if (!uri.scheme.empty()) {
      // Build cache key from URI
      cache_key = uri.address + "/" + uri.bucket_name;

      // Check cache first
      auto cached_fs = cache_.get(cache_key);
      if (cached_fs.has_value()) {
        return cached_fs.value();
      }

      // Cache miss - extract extfs.* properties and search for match
      ARROW_ASSIGN_OR_RAISE(auto external_fs_props_map, ExtractExternalFsProperties(properties));

      for (const auto& [fs_alias, fs_props] : external_fs_props_map) {
        auto address_result = api::GetValue<std::string>(fs_props, PROPERTY_FS_ADDRESS);
        auto bucket_result = api::GetValue<std::string>(fs_props, PROPERTY_FS_BUCKET_NAME);

        bool address_matches =
            uri.address.empty() || (address_result.ok() && address_result.ValueOrDie() == uri.address);
        bool bucket_matches = bucket_result.ok() && bucket_result.ValueOrDie() == uri.bucket_name;

        if (address_matches && bucket_matches) {
          ArrowFileSystemConfig config;
          auto status = CreateExternalFsConfig(fs_alias, fs_props, config);
          if (!status.ok()) {
            return arrow::Status::Invalid("Failed to create external filesystem config for '", fs_alias,
                                          "': ", status.ToString());
          }

          // Match by address (if specified) and bucket
          // Found matching external filesystem config - create and cache it
          ARROW_ASSIGN_OR_RAISE(auto fs, CreateArrowFileSystem(config));
          cache_.put(cache_key, fs);
          return fs;
        }
      }
      // No matching extfs.* config found, fall through to default
    }
  }

  // Create default filesystem from standard fs.* properties
  ArrowFileSystemConfig config;
  ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
  cache_key = config.GetCacheKey();

  // Check cache first
  auto cached_fs = cache_.get(cache_key);
  if (cached_fs.has_value()) {
    return cached_fs.value();
  }

  // Create and cache
  ARROW_ASSIGN_OR_RAISE(auto fs, CreateArrowFileSystem(config));
  cache_.put(cache_key, fs);
  return fs;
}

// ==================== StorageUri Implementation ====================

arrow::Result<StorageUri> StorageUri::Parse(const std::string& uri) {
  // Try to parse as a URI
  arrow::util::Uri parsed;
  auto status = parsed.Parse(uri);

  StorageUri result;

  // If parsing fails or scheme is empty, treat as relative path
  if (!status.ok() || parsed.scheme().empty()) {
    // Relative path - return as-is with empty scheme/address/bucket
    result.scheme = "";
    result.address = "";
    result.bucket_name = "";
    result.key = uri;
    return result;
  }

  // Successfully parsed as absolute URI with scheme
  result.scheme = parsed.scheme();
  result.address = parsed.host();  // Host is always the address/endpoint

  std::string path = parsed.path();

  // Path should contain bucket and key
  if (path.empty()) {
    return arrow::Status::Invalid("Storage URI missing bucket and key: ", uri);
  }

  // Remove leading slash if present
  if (path[0] == '/') {
    path = path.substr(1);
  }

  if (path.empty()) {
    return arrow::Status::Invalid("Storage URI missing bucket/container name: ", uri);
  }

  // Extract bucket/container from path (first component)
  size_t slash_pos = path.find('/');
  if (slash_pos == std::string::npos) {
    // Only bucket, no key
    result.bucket_name = path;
    result.key = "";
  } else {
    result.bucket_name = path.substr(0, slash_pos);
    result.key = path.substr(slash_pos + 1);
  }

  if (result.bucket_name.empty()) {
    return arrow::Status::Invalid("Missing bucket/container name in storage URI: ", uri);
  }

  return result;
}

// ==================== ArrowFileSystemSingleton Implementation ====================

void ArrowFileSystemSingleton::Init(const ArrowFileSystemConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (afs_ == nullptr) {
    auto result = CreateArrowFileSystem(config);
    if (!result.ok()) {
      throw std::runtime_error("Failed to init arrow filesystem: " + result.status().ToString());
    }
    afs_ = result.ValueOrDie();
  }
}

};  // namespace milvus_storage
