#pragma once

#include <arrow/type_fwd.h>
#include <vector>
#include <string>
#include "arrow/type.h"

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

std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types);
}  // namespace milvus_storage
