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

#include <unordered_set>
#include <string>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

static std::unordered_set<std::string> condition_write_key = {"If-None-Match", "x-goog-if-generation-match",
                                                              "x-cos-forbid-overwrite", "x-oss-forbid-overwrite"};

static std::unordered_map<std::string, std::pair<std::string, std::string>> condition_write_map = {
    {kCloudProviderAWS, {"If-None-Match", "*"}},
    {kCloudProviderGCP, {"x-goog-if-generation-match", "0"}},
    {kCloudProviderTencent, {"x-cos-forbid-overwrite", "true"}},
    {kCloudProviderAliyun, {"x-oss-forbid-overwrite", "true"}},
    {kAzureFileSystemName, {"If-None-Match", "*"}}};

bool IsConditionWriteKey(const std::string& key) { return condition_write_key.find(key) != condition_write_key.end(); }

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> open_condition_write_output_stream(
    const ArrowFileSystemPtr& fs, const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) {
  if (!metadata) {
    metadata = arrow::KeyValueMetadata::Make({}, {});
  }

  ARROW_ASSIGN_OR_RAISE(auto type_name, GetFileSystemTypeName(fs));

  if (auto it = condition_write_map.find(type_name); it != condition_write_map.end()) {
    metadata->Append(it->second.first, it->second.second);
  } else {  // Unsupported fs type
    return arrow::Status::NotImplemented("Conditional uploads are not supported for current fs type: ", type_name);
  }

  return fs->OpenOutputStream(path, metadata);
}

}  // namespace milvus_storage