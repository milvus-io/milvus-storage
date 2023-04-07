#pragma once

#include <string>
namespace milvus_storage {

class Status {
  public:
  static Status
  OK() {
    return Status(kOk);
  }

  static Status
  ArrowError(const std::string& msg) {
    return Status(kArrowError, msg);
  }

  std::string
  ToString() const;

  private:
  enum Code { kOk = 0, kArrowError = 1 };

  explicit Status(Code code, const std::string& msg = "") : code_(code), msg_(msg) {
  }

  Code code_;
  const std::string msg_;
};
}  // namespace milvus_storage