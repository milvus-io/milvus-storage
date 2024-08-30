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

#include "common/fs_util.h"
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "common/log.h"
#include "common/macro.h"

#ifdef MILVUS_OPENDAL
#endif

namespace milvus_storage {

Result<std::shared_ptr<arrow::fs::FileSystem>> BuildFileSystem(const std::string& uri, std::string* out_path) {
  arrow::util::Uri uri_parser;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  auto scheme = uri_parser.scheme();
  if (scheme == "file") {
    if (out_path == nullptr) {
      return Status::InvalidArgument("out_path should not be nullptr if scheme is file");
    }
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::LocalFileSystemOptions::FromUri(uri_parser, out_path));
    return std::shared_ptr<arrow::fs::FileSystem>(new arrow::fs::LocalFileSystem(option));
  } else if (scheme == "https") {
    if (!arrow::fs::IsS3Initialized()) {
      arrow::fs::S3GlobalOptions global_options;
      RETURN_ARROW_NOT_OK(arrow::fs::InitializeS3(global_options));
      std::atexit([]() {
        auto status = arrow::fs::EnsureS3Finalized();
        if (!status.ok()) {
          LOG_STORAGE_WARNING_ << "Failed to finalize S3: " << status.message();
        }
      });
    }
    arrow::fs::S3Options options;
    options.endpoint_override = uri_parser.ToString();
    uri_parser.password();
    options.ConfigureAccessKey(std::getenv("ACCESS_KEY"), std::getenv("SECRET_KEY"));
    *out_path = std::getenv("FILE_PATH");
    if (std::getenv("REGION") != nullptr) {
      options.region = std::getenv("REGION");
    }
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::S3FileSystem::Make(options));

    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }

  // if (schema == "hdfs") {
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto option, arrow::fs::HdfsOptions::FromUri(uri_parser));
  //   ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::HadoopFileSystem::Make(option));
  //   return std::shared_ptr<arrow::fs::FileSystem>(fs);
  // }
  return Status::InvalidArgument("Unsupported schema: " + scheme);
}
/**
 * Uri Convert to Path
 */
std::string UriToPath(const std::string& uri) {
  arrow::util::Uri uri_parser;
  auto status = uri_parser.Parse(uri);

  if (status.ok()) {
    return uri_parser.path();
  } else {
    return std::string("");
  }
}
};  // namespace milvus_storage
