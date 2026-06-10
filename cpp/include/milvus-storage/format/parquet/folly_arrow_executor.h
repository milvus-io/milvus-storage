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

#include <arrow/util/thread_pool.h>
#include <folly/Executor.h>

namespace milvus_storage::parquet {

// Adapt the Folly executor inherited from the consuming future chain for
// Arrow's async generator while retaining its KeepAlive for outstanding work.
std::shared_ptr<arrow::internal::Executor> MakeFollyArrowExecutor(folly::Executor::KeepAlive<> executor,
                                                                  int capacity = 1);

}  // namespace milvus_storage::parquet
