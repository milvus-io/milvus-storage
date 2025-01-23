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

#include "milvus-storage/reader/common/combine_reader.h"
#include <memory>
#include "arrow/type.h"
#include "milvus-storage/common/constants.h"
namespace milvus_storage {
std::unique_ptr<CombineReader> CombineReader::Make(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                                                   std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                                                   std::shared_ptr<Schema> schema) {
  assert(scalar_reader != nullptr && vector_reader != nullptr);
  return std::make_unique<CombineReader>(std::move(scalar_reader), std::move(vector_reader), schema);
}

std::shared_ptr<arrow::Schema> CombineReader::schema() const { return schema_->schema(); }

arrow::Status CombineReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> scalar_batch;
  std::shared_ptr<arrow::RecordBatch> vector_batch;
  ARROW_RETURN_NOT_OK(scalar_reader_->ReadNext(&scalar_batch));
  ARROW_RETURN_NOT_OK(vector_reader_->ReadNext(&vector_batch));
  if (scalar_batch == nullptr || vector_batch == nullptr) {
    batch = nullptr;
    return arrow::Status::OK();
  }

  for (int i = 0; i < scalar_batch->num_columns(); ++i) {
    if (scalar_batch->column_name(i) == kOffsetFieldName) {
      scalar_batch->RemoveColumn(i);
      break;
    }
  }

  assert(scalar_batch->num_rows() == vector_batch->num_rows());

  auto vec_column = vector_batch->GetColumnByName(schema_->options().vector_column);
  std::vector<std::shared_ptr<arrow::Array>> columns(scalar_batch->columns().begin(), scalar_batch->columns().end());

  auto vec_column_idx = schema_->schema()->GetFieldIndex(schema_->options().vector_column);
  columns.insert(columns.begin() + vec_column_idx, vec_column);

  *batch = arrow::RecordBatch::Make(schema(), scalar_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}

}  // namespace milvus_storage
