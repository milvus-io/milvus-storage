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
#include "filesystem/fs.h"
#include "filesystem/s3/s3_fs.h"
#include "filesystem/azure/azure_fs.h"

#ifdef MILVUS_OPENDAL
#endif

namespace milvus_storage {

Result<std::shared_ptr<arrow::fs::FileSystem>> FileSystemFactory::BuildFileSystem(const StorageConfig& storage_config,
                                                                                  std::string* out_path) {
  arrow::util::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(storage_config.uri));
  auto scheme = uri_parser.scheme();
  auto host = uri_parser.host();
  if (scheme == "file") {
    if (out_path == nullptr) {
      return Status::InvalidArgument("out_path should not be nullptr if scheme is file");
    }
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, out_path));
    return std::shared_ptr<arrow::fs::FileSystem>(new arrow::fs::LocalFileSystem(option));
  } else if (scheme == "https") {
    if (host.find("s3") != std::string::npos || host.find("googleapis") != std::string::npos ||
        host.find("oss") != std::string::npos || host.find("cos") != std::string::npos) {
      auto producer = std::make_shared<S3FileSystemProducer>();
      return producer->Make(storage_config, out_path);
    } else if (host.find("blob.core.windows.net") != std::string::npos) {
      auto producer = std::make_shared<AzureFileSystemProducer>();
      return producer->Make(storage_config, out_path);
    }
  }

  // if (schema == "hdfs") {
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::HdfsOptions::FromUri(uri_parser));
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::HadoopFileSystem::Make(option));
  //   return std::shared_ptr<arrow::fs::FileSystem>(fs);
  // }
  return Status::InvalidArgument("Unsupported schema: " + scheme);
};

};  // namespace milvus_storage
