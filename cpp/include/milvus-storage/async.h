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
#include <utility>

namespace milvus_storage {
struct AsyncStatus {
  enum Code { OK, Cancelled, Deadline, Overloaded, Busy, Memory, Exception, Arrow };
  Code code = OK;
  arrow::Status detail;
  AsyncStatus() = default;
  AsyncStatus(Code code) : code(code) {}
  AsyncStatus(arrow::Status status)
      : code(status.ok()              ? OK
             : status.IsOutOfMemory() ? Memory
             : status.IsCancelled()   ? Cancelled
                                      : Arrow),
        detail(std::move(status)) {}
  bool ok() const { return code == OK; }
};
class AsyncOperation {
  public:
  virtual ~AsyncOperation() = default;
  virtual void Cancel() = 0;
};
}  // namespace milvus_storage
