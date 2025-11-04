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

arrow::Result<ArrowFileSystemPtr> CreateArrowFileSystem(const ArrowFileSystemConfig& config) {
  std::string out_path;
  auto storage_type = StorageType_Map[config.storage_type];
  switch (storage_type) {
    case StorageType::Local: {
      arrow::util::Uri uri_parser;
      auto uri = "file://" + config.root_path;
      RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
      ASSIGN_OR_RETURN_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, &out_path));
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
          auto producer = std::make_shared<AzureFileSystemProducer>(config);
          return producer->Make();
        }
#endif
        case CloudProviderType::AWS:
        case CloudProviderType::GCP:
        case CloudProviderType::ALIYUN:
        case CloudProviderType::TENCENTCLOUD: {
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

arrow::Result<ArrowFileSystemPtr> ArrowFileSystemSingleton::createArrowFileSystem(const ArrowFileSystemConfig& config) {
  return CreateArrowFileSystem(config);
};

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

};  // namespace milvus_storage
