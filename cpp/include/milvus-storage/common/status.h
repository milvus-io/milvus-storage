// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

  bool ok() const { return code_ == kOk; }

  bool IsArrowError() const { return code_ == kArrowError; }

  bool IsInvalidArgument() const { return code_ == kInvalidArgument; }

  bool IsInternalStateError() const { return code_ == kInternalStateError; }

  bool IsFileNotFound() const { return code_ == kFileNotFound; }

  std::string ToString() const;

  private:
  enum Code {
    kOk = 0,
    kArrowError = 1,
    kInvalidArgument = 2,
    kInternalStateError = 3,
    kFileNotFound = 4,
  };

  explicit Status(Code code, const std::string& msg = "") : code_(code), msg_(msg) {}

  Code code_;
  std::string msg_;
};
}  // namespace milvus_storage
