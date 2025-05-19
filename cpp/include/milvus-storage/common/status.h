

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

  static Status FileNotFound(const std::string& msg = "") { return Status(kFileNotFound, msg); }

  static Status WriterError(const std::string& msg) { return Status(kWriterError, msg); }

  static Status ReaderError(const std::string& msg) { return Status(kReaderError, msg); }

  static Status MetadataParseError(const std::string& msg) { return Status(kMetadataParseError, msg); }

  static Status IOError(const std::string& msg) { return Status(kIOError, msg); }

  bool ok() const { return code_ == kOk; }

  bool IsArrowError() const { return code_ == kArrowError; }

  bool IsInvalidArgument() const { return code_ == kInvalidArgument; }

  bool IsInternalStateError() const { return code_ == kInternalStateError; }

  bool IsFileNotFound() const { return code_ == kFileNotFound; }

  bool IsWriterError() const { return code_ == kWriterError; }

  bool IsReaderError() const { return code_ == kReaderError; }

  bool IsIOError() const { return code_ == kIOError; }

  bool IsMetadataParseError() const { return code_ == kMetadataParseError; }

  std::string ToString() const;

  private:
  enum Code {
    kOk = 0,
    kArrowError = 1,
    kInvalidArgument = 2,
    kInternalStateError = 3,
    kFileNotFound = 4,
    kWriterError = 5,
    kIOError = 6,
    kReaderError = 7,
    kMetadataParseError = 8,
  };

  explicit Status(Code code, const std::string& msg = "") : code_(code), msg_(msg) {}

  Code code_;
  std::string msg_;
};
}  // namespace milvus_storage
