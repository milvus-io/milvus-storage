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

#include "milvus-storage/transaction/transaction.h"

#include <cassert>
#include <charconv>
#include <string_view>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/buffer.h>

#include "milvus-storage/common/lrucache.h"

namespace milvus_storage::api::transaction {

template <typename T>
TransactionImpl<T>::TransactionImpl(const api::Properties& properties, const std::string& base_path)
    : read_version_(MANIFEST_VERSION_INVALID),
      status_(TransStatus::STATUS_INIT),
      properties_(std::move(properties)),
      base_path_(base_path),
      handler_(nullptr),
      current_manifest_(nullptr) {}

template <typename T>
arrow::Status TransactionImpl<T>::begin(int64_t read_version) {
  if (status_ != TransStatus::STATUS_INIT) {
    return arrow::Status::Invalid("Transaction already begun. [status=", status_, "]");
  }

  ARROW_ASSIGN_OR_RAISE(auto handler_type, GetValue<std::string>(properties_, PROPERTY_TRANSACTION_HANDLER_TYPE));
  handler_ = TransactionHandler<T>::CreateTransactionHandler(handler_type, base_path_, properties_);
  assert(handler_);

  if (read_version <= MANIFEST_VERSION_INVALID) {
    ARROW_ASSIGN_OR_RAISE(read_version_, handler_->get_latest_version());
  } else {
    read_version_ = read_version;
  }

  ARROW_ASSIGN_OR_RAISE(current_manifest_, handler_->get_current_manifest(read_version_));

  // update the transaction state
  status_ = TransStatus::STATUS_BEGIN;
  return arrow::Status::OK();
}

template <typename T>
arrow::Result<std::shared_ptr<T>> TransactionImpl<T>::get_current_manifest() {
  if (status_ < TransStatus::STATUS_BEGIN) {
    return arrow::Status::Invalid("Transaction not begin. [status=", status_, "]");
  }
  return current_manifest_;
}

template <typename T>
arrow::Result<CommitResult> TransactionImpl<T>::commit(const std::shared_ptr<T>& new_manifest,
                                                       const UpdateType update_type,
                                                       const TransResolveStrategy resolve) {
  CommitResult commit_result;

  if (status_ != TransStatus::STATUS_BEGIN) {
    return arrow::Status::Invalid("Transaction not begin or already finished. [status=", status_, "]");
  }

  // Create the pending update then apply it
  // if apply failed, return COMMIT_CONFLICT
  auto update = PendingUpdate<T>::CreatePendingUpdate(update_type, current_manifest_, new_manifest);
  assert(update);
  auto applied_manifest_result = update->apply();
  if (!applied_manifest_result.ok()) {
    // quick return
    status_ = TransStatus::STATUS_ABORTED;
    return CommitResult{.success = false,
                        .read_version = read_version_,
                        .committed_version = MANIFEST_VERSION_INVALID,
                        .failed_message = "Failed to apply pending update to current manifest. [update_type=" +
                                          std::to_string(static_cast<int>(update_type)) +
                                          "], details: " + applied_manifest_result.status().ToString()};
  }
  auto applied_manifest = applied_manifest_result.ValueOrDie();

  switch (resolve) {
    case RESOLVE_FAIL: {
      // try to commit directly
      ARROW_ASSIGN_OR_RAISE(commit_result, handler_->commit(applied_manifest, read_version_, read_version_ + 1));
      break;
    }
    case RESOLVE_MERGE: {
      ARROW_ASSIGN_OR_RAISE(commit_result, handler_->commit(applied_manifest, read_version_, read_version_ + 1));
      // try to commit again, if current commit_result is true, won't enter the loop
      ARROW_ASSIGN_OR_RAISE(auto num_retries, GetValue<int32_t>(properties_, PROPERTY_TRANSACTION_COMMIT_NUM_RETRIES));
      while (!commit_result.success && num_retries > 0) {
        // reload the latest manifest and retry merge
        ARROW_ASSIGN_OR_RAISE(auto latest_version, handler_->get_latest_version());
        ARROW_ASSIGN_OR_RAISE(auto latest_manifest, handler_->get_current_manifest(latest_version));

        // recreate the pending update with latest manifest and apply
        update = PendingUpdate<T>::CreatePendingUpdate(update_type, latest_manifest, new_manifest);
        assert(update);
        applied_manifest_result = update->apply();
        if (!applied_manifest_result.ok()) {
          return CommitResult{
              .success = false,
              .read_version = read_version_,
              .committed_version = MANIFEST_VERSION_INVALID,
              .failed_message =
                  "Failed to apply pending update to latest manifest during merge resolve. [update_type=" +
                  std::to_string(static_cast<int>(update_type)) +
                  "], details: " + applied_manifest_result.status().ToString()};
        }
        applied_manifest = applied_manifest_result.ValueOrDie();
        ARROW_ASSIGN_OR_RAISE(commit_result, handler_->commit(applied_manifest, latest_version, latest_version + 1));
        num_retries--;
      }

      if (!commit_result.success) {
        // all retries failed
        commit_result.failed_message =
            "Exceeded maximum retry attempts for merge resolve strategy. " + commit_result.failed_message;
      }
      break;
    }
    case RESOLVE_OVERWRITE: {
      ARROW_ASSIGN_OR_RAISE(auto num_retries, GetValue<int32_t>(properties_, PROPERTY_TRANSACTION_COMMIT_NUM_RETRIES));
      do {
        ARROW_ASSIGN_OR_RAISE(auto latest_version, handler_->get_latest_version());
        ARROW_ASSIGN_OR_RAISE(commit_result, handler_->commit(applied_manifest, read_version_, latest_version + 1));
        num_retries--;
      } while (!commit_result.success && num_retries > 0);

      if (!commit_result.success) {
        // all retries failed
        commit_result.failed_message =
            "Exceeded maximum retry attempts for merge resolve strategy. " + commit_result.failed_message;
      }
      break;
    }
    default:
      return arrow::Status::Invalid("Unknown resolve strategy. [strategy=", resolve, "]");
  }

  // the execution will always REACH here after do resolver.
  status_ = commit_result.success ? TransStatus::STATUS_COMMITTED : TransStatus::STATUS_ABORTED;
  // always return the read version of current transaction
  commit_result.read_version = read_version_;
  return commit_result;
}

template <typename T>
arrow::Result<std::shared_ptr<T>> TransactionImpl<T>::get_latest_manifest() {
  assert(status_ == TransStatus::STATUS_INIT || status_ == TransStatus::STATUS_READ);
  if (status_ == TransStatus::STATUS_INIT) {
    ARROW_RETURN_NOT_OK(begin());
    status_ = TransStatus::STATUS_READ;
  }

  return current_manifest_;
}

template <typename T>
arrow::Status TransactionImpl<T>::abort() {
  if (status_ != TransStatus::STATUS_BEGIN) {
    return arrow::Status::Invalid("Transaction not begin or already finished. [status=", status_, "]");
  }

  status_ = TransStatus::STATUS_ABORTED;
  return arrow::Status::OK();
}

template <typename T>
int64_t TransactionImpl<T>::read_version() const {
  return read_version_;
}

template <typename T>
TransStatus TransactionImpl<T>::status() const {
  return status_;
}

template class TransactionImpl<Manifest>;

}  // namespace milvus_storage::api::transaction