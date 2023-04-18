#pragma once

#include <arrow/type_fwd.h>
#include <vector>
#include <string>
#include "arrow/type.h"

namespace milvus_storage {
std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types);
}  // namespace milvus_storage