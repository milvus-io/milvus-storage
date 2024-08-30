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

#pragma once
#include <arrow/filesystem/filesystem.h>
#include <memory>
#include <string>
#include "result.h"
namespace milvus_storage {

Result<std::shared_ptr<arrow::fs::FileSystem>> BuildFileSystem(const std::string& uri, std::string* out_path = nullptr);

std::string UriToPath(const std::string& uri);

static constexpr int64_t DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024;  // 1 MB

// https://github.com/apache/arrow/blob/6b268f62a8a172249ef35f093009c740c32e1f36/cpp/src/arrow/filesystem/s3fs.cc#L1596
static constexpr int64_t ARROW_PART_UPLOAD_SIZE = 10 * 1024 * 1024;  // 10 MB

static constexpr int64_t MIN_BUFFER_SIZE_PER_FILE = DEFAULT_MAX_ROW_GROUP_SIZE + ARROW_PART_UPLOAD_SIZE;

}  // namespace milvus_storage
