// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <exception>
#include <new>
#include <stdexcept>
#include <arrow/status.h>

namespace milvus_storage::detail {
// Keep diagnostic text at the business boundary; tracing exports only classes.
inline arrow::Status ExceptionStatus(const char* operation, const std::exception* error = nullptr) {
  if (error && dynamic_cast<const std::bad_alloc*>(error))
    return arrow::Status::OutOfMemory(operation, ": ", error->what());
  if (!error)
    return arrow::Status::UnknownError(operation, ": unidentified exception");
  if (dynamic_cast<const std::invalid_argument*>(error))
    return arrow::Status::Invalid(operation, ": ", error->what());
  return arrow::Status::UnknownError(operation, ": ", error->what());
}
}  // namespace milvus_storage::detail
