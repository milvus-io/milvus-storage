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

#include "reader/common/combine_reader.h"
#include "common/macro.h"
#include "arrow/type.h"
namespace milvus_storage {
Result<std::shared_ptr<CombineReader>> CombineReader::Make(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                                                           std::shared_ptr<arrow::RecordBatchReader> vector_reader,
                                                           std::shared_ptr<Schema> schema) {
  if (scalar_reader == nullptr || vector_reader == nullptr) {
    return Status::InvalidArgument("null reader");
  }
  return std::make_shared<CombineReader>(scalar_reader, vector_reader, schema);
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

  assert(scalar_batch->num_rows() == vector_batch->num_rows());

  auto vec_column = vector_batch->GetColumnByName(schema_->options()->vector_column);
  std::vector<std::shared_ptr<arrow::Array>> columns(scalar_batch->columns().begin(), scalar_batch->columns().end());

  auto vec_column_idx = schema_->schema()->GetFieldIndex(schema_->options()->vector_column);
  columns.insert(columns.begin() + vec_column_idx, vec_column);

  *batch = arrow::RecordBatch::Make(schema(), scalar_batch->num_rows(), std::move(columns));
  return arrow::Status::OK();
}
}  // namespace milvus_storage