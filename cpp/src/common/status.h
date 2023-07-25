#pragma once

#include <arrow/status.h>
#include <string>
#include "arrow/result.h"
namespace milvus_storage {

class Status {
  public:
  Status(const Status& s);

  Status& operator=(const Status& s);

  static Status OK() { return Status(kOk); }

  static Status ArrowError(const std::string& msg) { return Status(kArrowError, msg); }

  static Status InvalidArgument(const std::string& msg) { return Status(kInvalidArgument, msg); }

  static Status InternalStateError(const std::string& msg) { return Status(kInternalStateError, msg); }

  static Status ManifestNotFound(const std::string& msg = "") { return Status(kManifestNotFound, msg); }

  bool ok() const { return code_ == kOk; }

  bool IsArrowError() const { return code_ == kArrowError; }

  bool IsInvalidArgument() const { return code_ == kInvalidArgument; }

  bool IsInternalStateError() const { return code_ == kInternalStateError; }

  bool IsManifestNotFound() const { return code_ == kManifestNotFound; }

  std::string ToString() const;

  private:
  enum Code {
    kOk = 0,
    kArrowError = 1,
    kInvalidArgument = 2,
    kInternalStateError = 3,
    kManifestNotFound = 4,
  };

  explicit Status(Code code, const std::string& msg = "") : code_(code), msg_(msg) {}

  Code code_;
  std::string msg_;
};
}  // namespace milvus_storage
