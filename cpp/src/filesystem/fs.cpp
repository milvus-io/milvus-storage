

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
  switch (storage_type) {
    case StorageType::Local: {
      arrow::util::Uri uri_parser;
      auto uri = "file://" + config.root_path;
      RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
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
