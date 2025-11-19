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
#include <memory>

#include "milvus-storage/transaction/manifest.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::api::transaction {

struct CommitResult {
  bool success;
  int64_t read_version;
  int64_t committed_version;
  std::string failed_message;
};

template <typename T>
class TransactionHandler {
  public:
  virtual ~TransactionHandler() = default;

  virtual arrow::Result<int64_t> get_latest_version() = 0;

  virtual arrow::Result<std::shared_ptr<T>> get_current_manifest(int64_t version) = 0;

  // Commits the transaction with the provided new manifest.
  virtual arrow::Result<CommitResult> commit(std::shared_ptr<T>& manifest,
                                             int64_t old_version,
                                             int64_t new_version) = 0;

  static std::shared_ptr<TransactionHandler<T>> CreateTransactionHandler(const std::string& handler_type,
                                                                         const std::string& base_path,
                                                                         const api::Properties& properties);
};

}  // namespace milvus_storage::api::transaction