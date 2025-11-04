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

#include <cstdint>
#include <string>

#include "milvus-storage/properties.h"
#include "milvus-storage/transaction/manifest.h"
#include "milvus-storage/transaction/transhandler.h"
#include "milvus-storage/transaction/transupdate.h"

namespace milvus_storage::api::transaction {

enum TransStatus {
  STATUS_INIT = 0,       // initial state
  STATUS_BEGIN = 1,      // transaction has begun
  STATUS_COMMITTED = 2,  // transaction has been committed
  STATUS_ABORTED = 3,    // transaction has been aborted

  STATUS_READ = 101  // read status
};

enum TransResolveStrategy : int16_t {
  RESOLVE_FAIL = 0,
  RESOLVE_MERGE,

  TransResolveStrategyMax,
};

template <typename T>
class Transaction {
  public:
  virtual ~Transaction() = default;

  // begin the transaction
  virtual arrow::Status begin() = 0;

  // get the current manifest
  virtual arrow::Result<std::shared_ptr<T>> get_current_manifest() = 0;

  // commit with update manifest
  virtual arrow::Result<bool> commit(const std::shared_ptr<T>& new_manifest,
                                     const UpdateType update_type,
                                     const TransResolveStrategy resolve) = 0;

  // direct abort the transaction
  virtual arrow::Status abort() = 0;

  // direct get the latest manifest
  virtual arrow::Result<std::shared_ptr<T>> get_latest_manifest() = 0;

  // get current read version of manifest
  virtual int64_t read_version() const = 0;

  // get current status of transaction
  virtual TransStatus status() const = 0;
};  // class Transaction

template <typename T>
class TransactionImpl : public Transaction<T> {
  public:
  TransactionImpl(const api::Properties& properties, const std::string& base_path);
  ~TransactionImpl() override = default;

  arrow::Status begin() override;

  arrow::Result<std::shared_ptr<T>> get_current_manifest() override;

  arrow::Result<bool> commit(const std::shared_ptr<T>& new_manifest,
                             const UpdateType update_type,
                             const TransResolveStrategy resolve) override;

  arrow::Status abort() override;

  arrow::Result<std::shared_ptr<T>> get_latest_manifest() override;

  int64_t read_version() const override;

  TransStatus status() const override;

  private:
  TransStatus status_;
  int64_t read_version_;
  api::Properties properties_;
  std::string base_path_;

  std::shared_ptr<TransactionHandler<T>> handler_;
  std::shared_ptr<T> current_manifest_;
};

}  // namespace milvus_storage::api::transaction