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
  std::string uri = "";
  std::string storage_type = "local";  // [local, remote]
  std::string bucket_name = "";
  std::string access_key_id = "";
  std::string access_key_value = "";
  std::string cloud_provider = "";  // [aws, azure, gcp, tencent, aliyun]
  std::string region = "";
  bool use_custom_part_upload = false;

  std::string ToString() const {
    std::stringstream ss;
    ss << "[uri=" << uri << ", storage_type=" << storage_type << ", bucket_name=" << bucket_name
       << ", cloud_provider=" << cloud_provider << ", region=" << region
       << ", use_custom_part_upload=" << use_custom_part_upload << "]";

    return ss.str();
  }
};

class FileSystemProducer {
  public:
  virtual ~FileSystemProducer() = default;

  virtual Result<ArrowFileSystemPtr> Make(const ArrowFileSystemConfig& config, std::string* out_path) = 0;

  std::string UriToPath(const std::string& uri) {
    arrow::util::Uri uri_parser;
    auto status = uri_parser.Parse(uri);
    if (status.ok()) {
      return uri_parser.path();
    } else {
      return std::string("");
    }
  }
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
      afs_ = createArrowFileSystem(config).value();
    }
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    afs_.reset();
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
