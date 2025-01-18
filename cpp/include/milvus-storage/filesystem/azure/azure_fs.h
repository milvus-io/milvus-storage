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
#ifdef MILVUS_AZURE_FS

#include "arrow/filesystem/azurefs.h"
#include <cstdlib>
#include "common/log.h"
#include "common/macro.h"
#include "filesystem/fs.h"

namespace milvus_storage {

class AzureFileSystemProducer : public FileSystemProducer {
  public:
  AzureFileSystemProducer(){};

  Result<std::shared_ptr<arrow::fs::FileSystem>> Make(const StorageConfig& storage_config,
                                                      std::string* out_path) override {
    arrow::util::Uri uri_parser;
    RETURN_ARROW_NOT_OK(uri_parser.Parse(storage_config.uri));

    arrow::fs::AzureOptions options;
    auto account = storage_config.access_key_id;
    auto key = storage_config.access_key_value;
    if (account.empty() || key.empty()) {
      return Status::InvalidArgument("Please provide azure storage account and azure secret key");
    }
    options.account_name = account;
    RETURN_ARROW_NOT_OK(options.ConfigureAccountKeyCredential(key));

    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::AzureFileSystem::Make(options));
    fs->CreateDir(*out_path);
    return std::shared_ptr<arrow::fs::FileSystem>(fs);
  }
};

}  // namespace milvus_storage
#endif
