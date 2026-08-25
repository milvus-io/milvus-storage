// Copyright 2024 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <utility>

#include <arrow/status.h>

namespace milvus_storage {

/// Thread-safe first-failure state for writers with asynchronous work.
class WriterStatus {
  public:
  [[nodiscard]] arrow::Status Check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
  }

  [[nodiscard]] arrow::Status RecordFirstFailure(arrow::Status status) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status.ok() && status_.ok()) {
      status_ = std::move(status);
    }
    return status_.ok() ? status : status_;
  }

  private:
  mutable std::mutex mutex_;
  arrow::Status status_ = arrow::Status::OK();
};

}  // namespace milvus_storage
