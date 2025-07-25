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

#include <sstream>
#include <parquet/properties.h>

namespace milvus_storage {

static constexpr int64_t DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024;  // 1 MB

// https://github.com/apache/arrow/blob/6b268f62a8a172249ef35f093009c740c32e1f36/cpp/src/arrow/filesystem/s3fs.cc#L1596
static constexpr int64_t ARROW_PART_UPLOAD_SIZE = 10 * 1024 * 1024;  // 10 MB

static constexpr int64_t MIN_BUFFER_SIZE_PER_FILE = DEFAULT_MAX_ROW_GROUP_SIZE + ARROW_PART_UPLOAD_SIZE;

// Default number of rows to read when using ::arrow::RecordBatchReader
static constexpr int64_t DEFAULT_READ_BATCH_SIZE = 1024;
static constexpr int64_t DEFAULT_READ_BUFFER_SIZE = 16 * 1024 * 1024;   // 16 MB
static constexpr int64_t DEFAULT_WRITE_BUFFER_SIZE = 16 * 1024 * 1024;  // 16 MB

struct StorageConfig {
  int64_t part_size = 0;
};

}  // namespace milvus_storage
