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

#include <memory>

#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage::api::transaction {

enum UpdateType : int16_t {
  APPENDFILES = 0,
  ADDFIELD,

  UpdateTypeMax,
};

template <typename T>
class PendingUpdate {
  public:
  virtual ~PendingUpdate() = default;

  // apply the pending update and return the updated manifest
  // arrow::result will be error if the update fails
  virtual arrow::Result<std::shared_ptr<T>> apply() = 0;

  static std::shared_ptr<PendingUpdate<T>> CreatePendingUpdate(const UpdateType update_type,
                                                               const std::shared_ptr<T>& base_cgs,
                                                               const std::shared_ptr<T>& new_cg);
};

};  // namespace milvus_storage::api::transaction