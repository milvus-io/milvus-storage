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

#include <arrow/filesystem/localfs.h>
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "milvus-storage/common/path_util.h"
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/operations.hpp>

#ifdef MILVUS_AZURE_FS
#include "milvus-storage/filesystem/azure/azure_fs.h"
#endif

#ifdef MILVUS_OPENDAL
#endif

namespace milvus_storage {

Result<ArrowFileSystemPtr> ArrowFileSystemSingleton::createArrowFileSystem(const ArrowFileSystemConfig& config) {
  std::string out_path;
  auto storage_type = StorageType_Map[config.storage_type];
  arrow::util::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(config.address));
  switch (storage_type) {
    case StorageType::Local: {
      ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, &out_path));
      boost::filesystem::path dir_path(out_path);
      if (!boost::filesystem::exists(dir_path)) {
        boost::filesystem::create_directories(dir_path);
      }
      return ArrowFileSystemPtr(new arrow::fs::LocalFileSystem(option));
    }
    case StorageType::Remote: {
      auto cloud_provider = CloudProviderType_Map[config.cloud_provider];
      switch (cloud_provider) {
#ifdef MILVUS_AZURE_FS
        case CloudProviderType::AZURE: {
          auto producer = std::make_shared<AzureFileSystemProducer>();
          return producer->Make(config, &out_path);
        }
#endif
        case CloudProviderType::AWS: {
          auto producer = std::make_shared<AwsFileSystemProducer>();
          return producer->Make(config, &out_path);
        }
        case CloudProviderType::GCP: {
          auto producer = std::make_shared<GcpFileSystemProducer>();
          return producer->Make(config, &out_path);
        }
        case CloudProviderType::ALIYUN: {
          auto producer = std::make_shared<AliyunFileSystemProducer>();
          return producer->Make(config, &out_path);
        }
        case CloudProviderType::TENCENTCLOUD: {
          auto producer = std::make_shared<TencentCloudFileSystemProducer>();
          return producer->Make(config, &out_path);
        }
        default: {
          return Status::InvalidArgument("Unsupported cloud provider: " + config.cloud_provider);
        }
      }
    }
    default: {
      return Status::InvalidArgument("Unsupported storage type: " + config.storage_type);
    }
  }
};

};  // namespace milvus_storage
