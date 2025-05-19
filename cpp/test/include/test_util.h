

#pragma once

#include <arrow/type_fwd.h>
#include <vector>
#include <string>
#include "arrow/type.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/status.h"
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

Status PrepareSimpleParquetFile(std::shared_ptr<arrow::Schema> schema,
                                std::shared_ptr<arrow::fs::FileSystem> fs,
                                const std::string& file_path,
                                int num_rows);
}  // namespace milvus_storage
