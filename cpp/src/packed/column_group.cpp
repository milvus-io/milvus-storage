// Copyright 2024 Zilliz
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

#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/extend_status.h"
#include <arrow/table.h>
#include <arrow/status.h>

namespace milvus_storage {

ColumnGroup::ColumnGroup(GroupId group_id) : group_id_(group_id), memory_usage_(0) {}

ColumnGroup::ColumnGroup(GroupId group_id, const std::shared_ptr<arrow::RecordBatch>& batch)
    : group_id_(group_id), memory_usage_(0) {
  // A constructor has no status channel (AddRecordBatch rejects null with
  // PackedInvalidArgs); a null batch yields an empty group instead of the
  // previous null dereference below.
  if (batch == nullptr) {
    return;
  }
  batches_.emplace_back(batch);
  auto batch_size = GetRecordBatchMemorySize(batch);
  memory_usage_ += batch_size;
  // Keep the same bookkeeping as AddRecordBatch: without these, size()==1
  // paired with empty per-batch usages / a stale row count.
  batch_memory_usage_.push_back(batch_size);
  total_rows_ += batch->num_rows();
}

arrow::Status ColumnGroup::AddRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
  if (!batch) {
    return MakeExtendError(ExtendStatusCode::PackedInvalidArgs, "ColumnGroup::AddRecordBatch: batch is null");
  }
  batches_.emplace_back(batch);

  // update stats
  total_rows_ += batch->num_rows();
  size_t batch_memory_usage = GetRecordBatchMemorySize(batch);
  batch_memory_usage_.emplace_back(batch_memory_usage);
  memory_usage_ += batch_memory_usage;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroup::Table() const {
  auto result = arrow::Table::FromRecordBatches(batches_);
  if (!result.ok()) {
    // Keep the original StatusCode and detail; only the message gains context.
    //
    // The reason is no longer the one this comment used to give. An
    // unclassified Invalid no longer becomes DataFormatBroken -- both this and
    // an internal-invariant wrap reaches segcore as UnexpectedError, so there
    // is no storage classification to preserve on that axis. What is still worth preserving
    // is the arrow StatusCode itself: callers branch on IsIOError, and
    // rewriting a pre-IO schema mismatch into an IO failure would be a lie
    // about what happened.
    return result.status().WithMessage("ColumnGroup::Table: failed to merge record batches: ",
                                       result.status().message());
  }
  return result;
}

std::shared_ptr<arrow::Schema> ColumnGroup::Schema() const {
  // All batches in a group share one schema; avoid materializing the merged
  // table (which can fail) just to read it.
  return batches_.empty() ? nullptr : batches_.front()->schema();
}

std::shared_ptr<arrow::RecordBatch> ColumnGroup::GetRecordBatch(size_t index) const { return batches_[index]; }

size_t ColumnGroup::GetMemoryUsage() const { return memory_usage_; }

}  // namespace milvus_storage
