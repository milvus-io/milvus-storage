#pragma once

#include <arrow/type_fwd.h>

#include "../options/options.h"
class Schema {
 public:
  Schema(std::shared_ptr<arrow::Schema> &arrow_schema);
  bool is_vector_schema(SpaceOption *option);
  bool is_scalar_schema(SpaceOption *option);

 private:
  std::shared_ptr<arrow::Schema> &arrow_schema_;
};