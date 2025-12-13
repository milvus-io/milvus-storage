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

#pragma once

#include <memory>
#include <string>
#include <mutex>
#include <map>
#include <shared_mutex>
#include <unordered_map>
#include <functional>
#include <list>

#include <arrow/filesystem/filesystem.h>
#include <arrow/util/uri.h>
#include <arrow/result.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/properties.h"

namespace milvus_storage {

/**
 * @brief Parsed storage URI components
 *
 * A generic URI parser for cloud storage systems that supports:
 * - Absolute URIs: s3://[endpoint/]bucket/key
 * - Relative paths: path/to/file
 *
 * For absolute URIs with a scheme (e.g., s3://), the parser extracts:
 * - scheme: The URI scheme (e.g., "s3")
 * - address: Optional endpoint/host from the URI
 * - bucket_name: The bucket/container name (first path component)
 * - key: The object key/path within the bucket (remaining path components)
 *
 * For relative paths (no scheme), the parser returns:
 * - scheme: "" (empty)
 * - address: "" (empty)
 * - bucket_name: "" (empty)
 * - key: The full relative path
 */
struct StorageUri {
  std::string scheme;       // URI scheme (e.g., "s3"), or "" for relative paths
  std::string address;      // Optional endpoint/host, or "" for relative paths
  std::string bucket_name;  // Bucket/container name, or "" for relative paths
  std::string key;          // Object key/path, or full path for relative paths

  /**
   * @brief Parse a storage URI or relative path into components
   *
   * Supports multiple formats:
   * - scheme://bucket/key (bucket-only format with scheme)
   * - scheme://endpoint/bucket/key (with explicit endpoint)
   * - relative/path/to/file (relative path, no scheme)
   *
   * @param uri The URI string or relative path to parse
   * @return Result containing parsed StorageUri (always succeeds for relative paths)
   */
  static arrow::Result<StorageUri> Parse(const std::string& uri);
};

using ArrowFileSystemPtr = std::shared_ptr<arrow::fs::FileSystem>;

// TODO: it's not `arrow` namespace, we should change this struct name.
// TODO: after chunkmanager(in milvus) removed, we can remove the used key in storage
struct ArrowFileSystemConfig {
  std::string address = "localhost:9000";
  std::string bucket_name = "a-bucket";
  std::string access_key_id = "minioadmin";
  std::string access_key_value = "minioadmin";
  std::string root_path = "files";
  std::string storage_type = "local";
  std::string cloud_provider = "aws";
  [[maybe_unused]] std::string iam_endpoint = "";
  std::string log_level = "warn";  // only use on global config
  std::string region = "";
  bool use_ssl = false;
  std::string ssl_ca_cert = "";
  bool use_iam = false;
  bool use_virtual_host = false;
  int64_t request_timeout_ms = 3000;
  [[maybe_unused]] bool gcp_native_without_auth = false;
  [[maybe_unused]] std::string gcp_credential_json = "";
  [[maybe_unused]] bool use_custom_part_upload = true;
  uint32_t max_connections = 100;

  // Alias for external filesystem identification (e.g., "prod", "backup")
  // Empty for default filesystem
  std::string alias = "";

  static arrow::Status create_file_system_config(const milvus_storage::api::Properties& properties_map,
                                                 ArrowFileSystemConfig& result);

  /**
   * @brief Get the cache key for this filesystem configuration
   *
   * The cache key is the combination of address and bucket_name, which uniquely
   * identifies a filesystem endpoint and bucket combination.
   *
   * @return String in format "address/bucket_name"
   */
  [[nodiscard]] std::string GetCacheKey() const {
    if (alias.empty()) {
      return "";
    }
    return address + "/" + bucket_name;
  }

  [[nodiscard]] std::string ToString() const {
    std::stringstream ss;
    ss << "[address=" << address << ", bucket_name=" << bucket_name << ", root_path=" << root_path
       << ", storage_type=" << storage_type << ", cloud_provider=" << cloud_provider << ", log_level=" << log_level
       << ", region=" << region << ", use_ssl=" << std::boolalpha << use_ssl
       << ", ssl_ca_cert_length=" << ssl_ca_cert.size()  // only print cert length
       << ", use_iam=" << std::boolalpha << use_iam << ", use_virtual_host=" << std::boolalpha << use_virtual_host
       << ", request_timeout_ms=" << request_timeout_ms << ", max_connections=" << max_connections;
    if (!alias.empty()) {
      ss << ", alias=" << alias;
    }
    ss << "]";

    return ss.str();
  }
};

arrow::Result<ArrowFileSystemPtr> CreateArrowFileSystem(const ArrowFileSystemConfig& config);

class FileSystemProducer {
  public:
  virtual ~FileSystemProducer() = default;

  virtual arrow::Result<ArrowFileSystemPtr> Make() = 0;
};

/**
 * @brief Unified filesystem cache and external filesystem registry
 *
 * This singleton manages both:
 * 1. LRU cache of filesystem instances (by address+bucket combination)
 * 2. External filesystem configurations from properties (extfs.* entries)
 *
 * Thread-safe for concurrent access.
 */
class FilesystemCache {
  public:
  static FilesystemCache& getInstance();

  /**
   * @brief Get or create a filesystem from properties and path
   *
   * This is the main API for obtaining filesystems. It automatically:
   * - Resolves external filesystems (extfs.*) if path has a scheme
   * - Falls back to default filesystem (fs.*) otherwise
   * - Caches all filesystems by address+bucket
   *
   * @param properties Properties containing filesystem configuration
   * @param path Optional path to determine filesystem (empty = default filesystem)
   * @return Result containing the cached or newly created filesystem
   */
  [[nodiscard]] arrow::Result<ArrowFileSystemPtr> get(const api::Properties& properties, const std::string& path = "");

  /**
   * @brief Get the size of cached filesystems
   */
  [[nodiscard]] size_t size() const;

  /**
   * @brief Remove a cached filesystem by key
   */
  void remove(const std::string& key);

  /**
   * @brief Clear all cached filesystems and external filesystem registrations
   */
  void clean();

  /**
   * @brief Set the cache capacity
   */
  void set_capacity(size_t capacity);

  public:
  FilesystemCache(const FilesystemCache&) = delete;
  FilesystemCache& operator=(const FilesystemCache&) = delete;

  private:
  explicit FilesystemCache(size_t capacity) : cache_(capacity) {}
  ~FilesystemCache() = default;

  // Filesystem instance cache (LRU)
  LRUCache<std::string, ArrowFileSystemPtr> cache_;
};

// ArrowFileSystemSingleton used on milvus which won't change filesystem config
class ArrowFileSystemSingleton {
  private:
  ArrowFileSystemSingleton(){};

  public:
  ArrowFileSystemSingleton(const ArrowFileSystemSingleton&) = delete;
  ArrowFileSystemSingleton& operator=(const ArrowFileSystemSingleton&) = delete;

  static ArrowFileSystemSingleton& GetInstance() {
    static ArrowFileSystemSingleton instance;
    return instance;
  }

  void Init(const ArrowFileSystemConfig& config);

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (afs_ != nullptr) {
      afs_.reset();
      afs_ = nullptr;
    }
  }

  ArrowFileSystemPtr GetArrowFileSystem() {
    std::lock_guard<std::mutex> lock(mutex_);
    return afs_;
  }

  private:
  ArrowFileSystemPtr afs_ = nullptr;
  std::mutex mutex_;
};

enum class StorageType {
  None = 0,
  Local = 1,
  Minio = 2,
  Remote = 3,
};

enum class CloudProviderType : int8_t {
  UNKNOWN = 0,
  AWS = 1,
  GCP = 2,
  ALIYUN = 3,
  AZURE = 4,
  TENCENTCLOUD = 5,
  HUAWEICLOUD = 7,
};

static std::map<std::string, StorageType> StorageType_Map = {{"local", StorageType::Local},
                                                             {"remote", StorageType::Remote}};

static std::map<std::string, CloudProviderType> CloudProviderType_Map = {{"aws", CloudProviderType::AWS},
                                                                         {"gcp", CloudProviderType::GCP},
                                                                         {"aliyun", CloudProviderType::ALIYUN},
                                                                         {"azure", CloudProviderType::AZURE},
                                                                         {"tencent", CloudProviderType::TENCENTCLOUD},
                                                                         {"huawei", CloudProviderType::HUAWEICLOUD}};

}  // namespace milvus_storage
