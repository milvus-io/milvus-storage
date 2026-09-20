// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <string>
#include <arrow/result.h>

namespace milvus_storage {
// C ABI operation admission only; synchronous filesystem memory policy is unchanged.
struct AsyncManifestLimits {
  static arrow::Result<size_t> Setting(const char* name, size_t default_value, size_t maximum) {
    const char* value = std::getenv(name);
    if (!value)
      return default_value;
    size_t result = 0;
    auto end = value + std::strlen(value);
    auto parsed = std::from_chars(value, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end || !result || result > maximum)
      return arrow::Status::Invalid("Invalid async runtime setting: ", name);
    return result;
  }
  size_t max_operations;
  static const arrow::Result<AsyncManifestLimits>& Get() {
    static const auto limits = []() -> arrow::Result<AsyncManifestLimits> {
      ARROW_ASSIGN_OR_RAISE(auto operations, Setting("LOON_ASYNC_MAX_OPERATIONS", 256, 65536));
      return AsyncManifestLimits{operations};
    }();
    return limits;
  }
};
}  // namespace milvus_storage
