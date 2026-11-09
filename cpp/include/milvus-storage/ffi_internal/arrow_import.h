// Copyright 2026 Zilliz
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

#include <arrow/c/abi.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>

namespace milvus_storage {

// Import each top-level column with its own C Data owner, without copying its
// buffers. A cached column must not retain the other columns of the input batch.
// Consumes array on both success and failure; callers must not release it again.
arrow::Result<std::shared_ptr<arrow::RecordBatch>> ImportWriterBatch(ArrowArray* array,
                                                                     const std::shared_ptr<arrow::Schema>& schema);

}  // namespace milvus_storage
