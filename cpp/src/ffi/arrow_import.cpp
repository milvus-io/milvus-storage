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

#include "milvus-storage/ffi_internal/arrow_import.h"

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/c/helpers.h>
#include <arrow/util/bitmap_ops.h>

#include <vector>

namespace milvus_storage {
namespace {
struct OwnedArrowArray {
  ArrowArray array{};
  OwnedArrowArray() = default;
  OwnedArrowArray(const OwnedArrowArray&) = delete;
  OwnedArrowArray& operator=(const OwnedArrowArray&) = delete;
  ~OwnedArrowArray() { ArrowArrayRelease(&array); }
};
}  // namespace

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ImportWriterBatch(ArrowArray* array,
                                                                     const std::shared_ptr<arrow::Schema>& schema) {
  if (ArrowArrayIsReleased(array)) {
    return arrow::Status::Invalid("Cannot import released ArrowArray");
  }
  OwnedArrowArray root;
  ArrowArrayMove(array, &root.array);
  const auto& batch = root.array;
  if (batch.length < 0 || batch.offset != 0 || batch.n_buffers != 1 || !batch.buffers || batch.dictionary ||
      batch.n_children != schema->num_fields() || (batch.n_children > 0 && !batch.children)) {
    return arrow::Status::Invalid("Invalid ArrowArray record batch layout");
  }
  auto null_count = batch.null_count;
  if (null_count == -1) {
    null_count = batch.buffers[0] ? batch.length - arrow::internal::CountSetBits(
                                                       static_cast<const uint8_t*>(batch.buffers[0]), 0, batch.length)
                                  : 0;
  }
  if (null_count != 0) {
    return arrow::Status::Invalid("ArrowArray record batch must not contain null rows");
  }
  const auto length = batch.length;
  for (int i = 0; i < schema->num_fields(); ++i) {
    if (!batch.children[i] || ArrowArrayIsReleased(batch.children[i]) || batch.children[i]->length < length) {
      return arrow::Status::Invalid("Invalid ArrowArray record batch child at index ", i);
    }
  }

  std::vector<OwnedArrowArray> children(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    ArrowArrayMove(batch.children[i], &children[i].array);
  }
  // Release the parent after moving its children. Each moved child now owns
  // its own release callback, independently of the other columns.
  ArrowArrayRelease(&root.array);

  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto column, arrow::ImportArray(&children[i].array, schema->field(i)->type()));
    columns.emplace_back(column->Slice(0, length));
  }
  auto record = arrow::RecordBatch::Make(schema, length, std::move(columns));
  ARROW_RETURN_NOT_OK(record->Validate());
  return record;
}

}  // namespace milvus_storage
