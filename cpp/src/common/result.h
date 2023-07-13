#pragma once
#include "assert.h"
#include "status.h"
#include <arrow/result.h>
#include <optional>

namespace milvus_storage {

template <typename T>
class Result {
  public:
  Result(T& value) noexcept : value_(value) {}  // NOLINT
  Result(T&& value) noexcept                    // NOLINT
      : value_(std::move(value)) {}

  Result(const Status& status) noexcept  // NOLINT
      : status_(status) {}

  Result(Status&& status) noexcept  // NOLINT
      : status_(std::move(status)) {}

  Result(const Result& result) {
    if (result.value_.has_value()) {
      value_ = result.value_.value();
    }
    if (result.status_.has_value()) {
      status_ = result.status_.value();
    }
  }

  Result(Result&& result) noexcept {
    if (result.value_.has_value()) {
      value_ = std::move(result.value_.value());
    }
    if (result.status_.has_value()) {
      status_ = std::move(result.status_.value());
    }
  }

  Result& operator=(const Result& result) {
    if (result.value_.has_value()) {
      value_ = result.value_.value();
    }
    if (result.status_.has_value()) {
      status_ = result.status_.value();
    }
    return *this;
  }

  Result& operator=(Result&& result) noexcept {
    if (result.value_.has_value()) {
      value_ = std::move(result.value_.value());
    }
    if (result.status_.has_value()) {
      status_ = std::move(result.status_.value());
    }
    return *this;
  }

  ~Result() = default;

  bool ok() { return value_.has_value(); }

  bool has_value() { return value_.has_value(); }

  T& value() & {
    assert(value_.has_value());
    return value_.value();
  }

  T value() && {
    assert(value_.has_value());
    return std::move(value_.value());
  }

  Status& status() {
    if (!status_.has_value() && value_.has_value()) {
      status_ = Status::OK();
    }
    assert(status_.has_value());
    return status_.value();
  }

  template <typename E>
  static Result<E> FromArrowResult(arrow::Result<E> arrow_result);

  private:
  std::optional<T> value_;
  std::optional<Status> status_;
};
}  // namespace milvus_storage
