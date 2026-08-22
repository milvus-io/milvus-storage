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

/// First-failure state shared by stateful writers.
///
/// A writer cannot safely resume after an operation fails: format encoders,
/// output streams, offsets, and buffered data may already have advanced by
/// different amounts. Every later operation returns the first failure unchanged.
///
/// Allocation failure is deliberately NOT handled here. Recording a failure
/// copies an arrow::Status, which allocates; if that allocation fails the
/// exception propagates and the process dies. A writer that cannot even record
/// why it failed has nothing useful left to do, and the machinery needed to
/// keep it "working" is what previously left callers waiting on a future that
/// nobody could ever complete. Crashing is recoverable; hanging is not.
class WriterStatus {
  public:
  WriterStatus() = default;
  WriterStatus(const WriterStatus&) = delete;
  WriterStatus& operator=(const WriterStatus&) = delete;

  // Moving an object requires exclusive ownership just like moving any other
  // C++ object. Do not lock here: every Status move is noexcept, and the flag
  // keeps the source terminal without allocating a replacement Status.
  WriterStatus(WriterStatus&& other) noexcept
      : status_(std::move(other.status_)), moved_from_(std::exchange(other.moved_from_, true)) {}

  WriterStatus& operator=(WriterStatus&& other) noexcept {
    if (this != &other) {
      status_ = std::move(other.status_);
      moved_from_ = std::exchange(other.moved_from_, true);
    }
    return *this;
  }

  [[nodiscard]] arrow::Status Check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (moved_from_) {
      return MovedFromStatus();
    }
    return status_;
  }

  [[nodiscard]] arrow::Status Fail(arrow::Status status) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (moved_from_) {
      return MovedFromStatus();
    }
    if (!status.ok() && status_.ok()) {
      status_ = std::move(status);
    }
    // A concurrent child may have poisoned the writer after the caller's
    // leading Check() but before this operation completed. Returning the
    // operation's local OK in that race would let a failed writer report one
    // more successful call. Once a first failure exists it is the answer to
    // every operation, including one whose local work happened to succeed.
    return status_.ok() ? status : status_;
  }

  // Record a failure observed by an asynchronous child operation immediately.
  // The aggregate future may not be complete yet, but the writer is already
  // unsafe to reuse: a later operation could otherwise append more data while
  // an earlier part/block has failed.
  void ObserveFailure(const arrow::Status& status) {
    if (status.ok()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (moved_from_ || !status_.ok()) {
      return;
    }
    status_ = status;
  }

  // Explicit Abort() makes a healthy writer terminal. Existing failures remain
  // untouched so cleanup can never hide the real cause.
  void BeginDiscard() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!moved_from_ && status_.ok()) {
      status_ = DiscardStatus();
    }
  }

  [[nodiscard]] bool ok() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !moved_from_ && status_.ok();
  }

  private:
  /// A moved-from writer state must not read as healthy.
  ///
  /// `arrow::Status`'s move leaves the source OK, so a WriterStatus that had
  /// recorded a failure came back ok() after being moved out of -- the one
  /// answer this class exists to never give. Using a moved-from object is a bug
  /// either way; this only decides whether that bug fails closed or lets a
  /// writer with a failed part accept more data.
  static arrow::Status MovedFromStatus() {
    return arrow::Status::Invalid("Stateful writer state was moved from and must not be used");
  }

  static arrow::Status DiscardStatus() { return arrow::Status::Cancelled("Stateful writer was discarded"); }

  mutable std::mutex mutex_;
  arrow::Status status_ = arrow::Status::OK();
  bool moved_from_ = false;
};

}  // namespace milvus_storage
