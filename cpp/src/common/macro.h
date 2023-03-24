#pragma once

#include <parquet/exception.h>
#define ASSIGN_OR_RETURN_NOT_OK(lhs, rexpr) \
  auto status_name = (rexpr);               \
  if (!status_name.ok()) {                  \
    return;                                 \
  }                                         \
  lhs = std::move(status_name).ValueOrDie();

#define RETURN_IGNORE_NOT_OK(status) \
  if (!status.ok()) {                \
    return;                          \
  }
