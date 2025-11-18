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

namespace milvus_storage {

// current config will be used in packed writer/reader layer
// TODO: will move it to properties.h in the future
static constexpr int64_t DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024;  // 1 MB

// Default number of rows to read when using ::arrow::RecordBatchReader
static constexpr int64_t DEFAULT_READ_BATCH_SIZE = 1024;
static constexpr int64_t DEFAULT_READ_BUFFER_SIZE = 16 * 1024 * 1024;   // 16 MB
static constexpr int64_t DEFAULT_WRITE_BUFFER_SIZE = 16 * 1024 * 1024;  // 16 MB

struct StorageConfig {
  int64_t part_size = 0;
};

#define LOON_FORMAT_PARQUET "parquet"
#define LOON_FORMAT_VORTEX "vortex"

#define ENCRYPTION_ALGORITHM_AES_GCM_V1 "AES_GCM_V1"
#define ENCRYPTION_ALGORITHM_AES_GCM_CTR_V1 "AES_GCM_CTR_V1"

#define LOON_COLUMN_GROUP_POLICY_SINGLE "single"
#define LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED "schema_based"
#define LOON_COLUMN_GROUP_POLICY_SIZE_BASED "size_based"

#define TRANSACTION_HANDLER_TYPE_UNSAFE "unsafe"
#define TRANSACTION_HANDLER_TYPE_DEFAULT TRANSACTION_HANDLER_TYPE_UNSAFE

}  // namespace milvus_storage
