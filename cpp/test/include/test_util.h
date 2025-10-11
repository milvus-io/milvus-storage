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

#include <arrow/type_fwd.h>
#include <arrow/status.h>
#include <vector>
#include <string>
#include "arrow/type.h"
#include "milvus-storage/common/macro.h"
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/properties.h"

namespace milvus_storage {
#define ASSERT_STATUS_OK(status)                       \
  do {                                                 \
    ASSERT_TRUE((status).ok()) << (status).ToString(); \
  } while (false)

#define ASSERT_STATUS_NOT_OK(status) \
  do {                               \
    ASSERT_FALSE((status).ok());     \
  } while (false)

#define ASSERT_AND_ASSIGN_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                           \
  ASSERT_STATUS_OK(status_name.status());               \
  lhs = std::move(status_name).ValueOrDie();

#define ASSERT_AND_ASSIGN(lhs, rexpr) ASSERT_AND_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);

std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types);

arrow::Status PrepareSimpleParquetFile(std::shared_ptr<arrow::Schema> schema,
                                       std::shared_ptr<arrow::fs::FileSystem> fs,
                                       const std::string& file_path,
                                       int num_rows);

// Helper method to get environment variable
std::string GetEnvVar(const std::string& var_name);

// Init common properties for tests
#define ENV_VAR_STORAGE_TYPE "STORAGE_TYPE"
#define ENV_VAR_ADDRESS "ADDRESS"
#define ENV_VAR_BUCKET_NAME "BUCKET_NAME"
#define ENV_VAR_ACCESS_KEY_ID "ACCESS_KEY"
#define ENV_VAR_ACCESS_KEY_VALUE "SECRET_KEY"
#define ENV_VAR_REGION "REGION"
#define ENV_VAR_ROOT_PATH "ROOT_PATH"

void InitTestProperties(api::Properties& properties, std::string root_path = "./");

}  // namespace milvus_storage
