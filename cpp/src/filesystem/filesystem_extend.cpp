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

#include "milvus-storage/filesystem/filesystem_extend.h"

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/filesystem/s3/s3_fs.h"

namespace milvus_storage {

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> open_condition_write_output_stream(
    const ArrowFileSystemPtr& fs, const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) {
  if (!metadata) {
    metadata = arrow::KeyValueMetadata::Make({}, {});
  }

  ARROW_ASSIGN_OR_RAISE(auto type_name, GetFileSystemTypeName(fs));

  if (type_name == kCloudProviderAWS) {
    metadata->Append("If-None-Match", "*");
  } else if (type_name == kCloudProviderGCP) {
    metadata->Append("x-goog-if-generation-match", "0");
  } else if (type_name == kCloudProviderTencent) {
    metadata->Append("x-cos-forbid-overwrite", "true");
  } else if (type_name == kCloudProviderAliyun) {
    metadata->Append("x-oss-forbid-overwrite", "true");
  } else if (type_name == kAzureFileSystemName) {
    metadata->Append("If-None-Match", "*");
  } else {  // Unsupported fs type
    return arrow::Status::NotImplemented("Conditional uploads are not supported for current fs type: ", type_name);
  }

  return fs->OpenOutputStream(path, metadata);
}

}  // namespace milvus_storage