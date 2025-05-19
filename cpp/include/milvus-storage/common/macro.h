

#pragma once

#include <parquet/exception.h>

namespace milvus_storage {

#define CONCAT_IMPL(x, y) x##y

#define CONCAT(x, y) CONCAT_IMPL(x, y)

#undef RETURN_NOT_OK
#define RETURN_NOT_OK(status) \
  do {                        \
    auto _s = (status);       \
    if (!_s.ok()) {           \
      return _s;              \
    }                         \
  } while (false)

#define RETURN_ARROW_NOT_OK(status)               \
  do {                                            \
    auto _s = (status);                           \
    if (!_s.ok()) {                               \
      return Status::ArrowError((_s).ToString()); \
    }                                             \
  } while (false)

#define RETURN_ARROW_NOT_OK_WITH_PREFIX(msg, staus)              \
  do {                                                           \
    auto status_name = (status);                                 \
    if (!status_name.ok()) {                                     \
      return Status::ArrowError((msg) + status_name.ToString()); \
    }                                                            \
  } while (false)

#define ASSIGN_OR_RETURN_ARROW_NOT_OK_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                                       \
  RETURN_ARROW_NOT_OK(status_name.status());                        \
  lhs = std::move(status_name).ValueOrDie();

#define ASSIGN_OR_RETURN_ARROW_NOT_OK(lhs, rexpr) \
  ASSIGN_OR_RETURN_ARROW_NOT_OK_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);

#define ASSIGN_OR_RETURN_NOT_OK_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                                 \
  RETURN_NOT_OK(status_name.status());                        \
  lhs = std::move(status_name).value();

#define ASSIGN_OR_RETURN_NOT_OK(lhs, rexpr) ASSIGN_OR_RETURN_NOT_OK_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);

}  // namespace milvus_storage
