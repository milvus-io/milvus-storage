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
#include "milvus-storage/properties.h"

namespace milvus_storage {

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

  static arrow::Status create_file_system_config(const milvus_storage::api::Properties& properties_map,
                                                 ArrowFileSystemConfig& result);

  bool operator<(const ArrowFileSystemConfig& other) const {
    return std::tie(address, bucket_name, access_key_id, access_key_value, root_path, storage_type, cloud_provider,
                    log_level, region, use_ssl, ssl_ca_cert, use_iam, use_virtual_host, request_timeout_ms,
                    max_connections) < std::tie(other.address, other.bucket_name, other.access_key_id,
                                                other.access_key_value, other.root_path, other.storage_type,
                                                other.cloud_provider, other.log_level, other.region, other.use_ssl,
                                                other.ssl_ca_cert, other.use_iam, other.use_virtual_host,
                                                other.request_timeout_ms, other.max_connections);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "[address=" << address << ", bucket_name=" << bucket_name << ", root_path=" << root_path
       << ", storage_type=" << storage_type << ", cloud_provider=" << cloud_provider << ", log_level=" << log_level
       << ", region=" << region << ", use_ssl=" << std::boolalpha << use_ssl
       << ", ssl_ca_cert=" << ssl_ca_cert.size()  // only print cert length
       << ", use_iam=" << std::boolalpha << use_iam << ", use_virtual_host=" << std::boolalpha << use_virtual_host
       << ", request_timeout_ms=" << request_timeout_ms << "]"
       << ", max_connections=" << max_connections << "]";

    return ss.str();
  }
};

class FileSystemProducer {
  public:
  virtual ~FileSystemProducer() = default;

  virtual arrow::Result<ArrowFileSystemPtr> Make() = 0;
};

arrow::Result<ArrowFileSystemPtr> CreateArrowFileSystem(const ArrowFileSystemConfig& config);

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

  void Init(const ArrowFileSystemConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (afs_ == nullptr) {
      auto result = createArrowFileSystem(config);
      if (!result.ok()) {
        throw std::runtime_error("Failed to init arrow filesystem: " + result.status().ToString());
      }
      afs_ = result.ValueOrDie();
    }
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (afs_ != nullptr) {
      afs_.reset();
    }
  }

  ArrowFileSystemPtr GetArrowFileSystem() {
    std::lock_guard<std::mutex> lock(mutex_);
    return afs_;
  }

  private:
  arrow::Result<ArrowFileSystemPtr> createArrowFileSystem(const ArrowFileSystemConfig& config);

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
};

static std::map<std::string, StorageType> StorageType_Map = {{"local", StorageType::Local},
                                                             {"remote", StorageType::Remote}};

static std::map<std::string, CloudProviderType> CloudProviderType_Map = {{"aws", CloudProviderType::AWS},
                                                                         {"gcp", CloudProviderType::GCP},
                                                                         {"aliyun", CloudProviderType::ALIYUN},
                                                                         {"azure", CloudProviderType::AZURE},
                                                                         {"tencent", CloudProviderType::TENCENTCLOUD}};

}  // namespace milvus_storage
