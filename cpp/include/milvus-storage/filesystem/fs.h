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

#include <arrow/filesystem/filesystem.h>
#include <arrow/util/uri.h>
#include <memory>
#include <string>
#include <mutex>
#include "milvus-storage/common/result.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

using ArrowFileSystemPtr = std::shared_ptr<arrow::fs::FileSystem>;

struct ArrowFileSystemConfig {
  std::string address = "localhost:9000";
  std::string bucket_name = "a-bucket";
  std::string access_key_id = "minioadmin";
  std::string access_key_value = "minioadmin";
  std::string root_path = "files";
  std::string storage_type = "minio";
  std::string cloud_provider = "aws";
  std::string iam_endpoint = "";
  std::string log_level = "warn";
  std::string region = "";
  bool useSSL = false;
  std::string sslCACert = "";
  bool useIAM = false;
  bool useVirtualHost = false;
  int64_t requestTimeoutMs = 3000;
  bool gcp_native_without_auth = false;
  std::string gcp_credential_json = "";
  bool use_custom_part_upload = true;

  std::string ToString() const {
    std::stringstream ss;
    ss << "[address=" << address << ", bucket_name=" << bucket_name << ", root_path=" << root_path
       << ", storage_type=" << storage_type << ", cloud_provider=" << cloud_provider
       << ", iam_endpoint=" << iam_endpoint << ", log_level=" << log_level << ", region=" << region
       << ", useSSL=" << std::boolalpha << useSSL << ", sslCACert=" << sslCACert.size()  // only print cert length
       << ", useIAM=" << std::boolalpha << useIAM << ", useVirtualHost=" << std::boolalpha << useVirtualHost
       << ", requestTimeoutMs=" << requestTimeoutMs << ", gcp_native_without_auth=" << std::boolalpha
       << gcp_native_without_auth << "]";

    return ss.str();
  }
};

class FileSystemProducer {
  public:
  virtual ~FileSystemProducer() = default;

  virtual Result<ArrowFileSystemPtr> Make() = 0;
};

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
      afs_ = result.value();
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
  Result<ArrowFileSystemPtr> createArrowFileSystem(const ArrowFileSystemConfig& config);

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
