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
#include <vector>
#include <string>
#include "arrow/type.h"
#include "common/macro.h"
#include "common/status.h"
#include "arrow/filesystem/filesystem.h"

namespace milvus_storage {
#define ASSERT_STATUS_OK(status) \
  do {                           \
    ASSERT_TRUE((status).ok());  \
  } while (false)

#define ASSERT_STATUS_NOT_OK(status) \
  do {                               \
    ASSERT_FALSE((status).ok());     \
  } while (false)

#define ASSERT_AND_ASSIGN_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                           \
  ASSERT_STATUS_OK(status_name.status());               \
  lhs = std::move(status_name).value();

#define ASSERT_AND_ASSIGN(lhs, rexpr) ASSERT_AND_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);

#define ASSERT_AND_ARROW_ASSIGN_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                                 \
  ASSERT_STATUS_OK(status_name.status());                     \
  lhs = std::move(status_name).ValueUnsafe();

#define ASSERT_AND_ARROW_ASSIGN(lhs, rexpr) ASSERT_AND_ARROW_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);
std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types);

Status PrepareSimpleParquetFile(arrow::fs::FileSystem& fs, const std::string& file_path, int num_rows);
}  // namespace milvus_storage
