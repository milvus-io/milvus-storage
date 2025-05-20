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

#ifdef MILVUS_AZURE_FS

#include "arrow/filesystem/azurefs.h"
#include <cstdlib>
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/azure/azure_fs.h"

namespace milvus_storage {

Result<ArrowFileSystemPtr> AzureFileSystemProducer::Make() {
  arrow::fs::AzureOptions options;
  arrow::util::Uri uri_parser;
  auto uri = options.blob_storage_scheme + "://" + config_.bucket_name + options.blob_storage_authority;
  RETURN_ARROW_NOT_OK(uri_parser.Parse(uri));
  assert(!config_.access_key_id.empty());
  options.account_name = config_.access_key_id;
  if (config_.useIAM) {
    assert(getenv("AZURE_CLIENT_ID") != NULL);
    assert(getenv("AZURE_TENANT_ID") != NULL);
    assert(getenv("AZURE_FEDERATED_TOKEN_FILE") != NULL);
    options.ConfigureWorkloadIdentityCredential();
  } else {
    // need azure secret key without iam
    assert(!config_.access_key_value.empty());
    RETURN_ARROW_NOT_OK(options.ConfigureAccountKeyCredential(config_.access_key_value));
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto fs, arrow::fs::AzureFileSystem::Make(options));
  fs->CreateDir(config_.root_path, true);
  return ArrowFileSystemPtr(fs);
}

}  // namespace milvus_storage
#endif
