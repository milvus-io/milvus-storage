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
#include "milvus-storage/common/result.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

class FileSystemProducer {
  public:
  virtual ~FileSystemProducer() = default;

  virtual Result<std::shared_ptr<arrow::fs::FileSystem>> Make(const StorageConfig& storage_config,
                                                              std::string* out_path) = 0;

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

class FileSystemFactory {
  public:
  Result<std::shared_ptr<arrow::fs::FileSystem>> BuildFileSystem(const StorageConfig& storage_config,
                                                                 std::string* out_path);
};

}  // namespace milvus_storage
