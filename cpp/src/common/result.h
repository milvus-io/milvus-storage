#pragma once
#include "status.h"
#include <optional>
#include "assert.h"

namespace milvus_storage {

template <typename T>
class Result {
  public:
  Result(const T&& value)  // NOLINT
      : value_(value) {
  }

  Result(const Status& status) noexcept  // NOLINT
      : status_(status) {
  }

  bool
  ok() {
    return value_.has_value();
  }

  T&
  value() {
    assert(value_.has_value());
    return value_.value();
  }

  private:
  std::optional<T> value_;
  std::optional<Status> status_;
};
}  // namespace milvus_storage