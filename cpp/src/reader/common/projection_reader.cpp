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

#include "reader/common/projection_reader.h"
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <algorithm>
#include <memory>
#include "common/macro.h"
#include "common/utils.h"

namespace milvus_storage {

ProjectionReader::ProjectionReader(std::shared_ptr<arrow::Schema> schema,
                                   std ::shared_ptr<arrow::RecordBatchReader> reader,
                                   std::shared_ptr<ReadOptions> options)
    : reader_(std::move(reader)), options_(std::move(options)), schema_(schema) {}

Result<std::shared_ptr<arrow::RecordBatchReader>> ProjectionReader::Make(
    std::shared_ptr<arrow::Schema> schema,
    std ::shared_ptr<arrow::RecordBatchReader> reader,
    std::shared_ptr<ReadOptions> options) {
  ASSIGN_OR_RETURN_NOT_OK(auto projection_schema, ProjectSchema(schema, options->columns));
  std::shared_ptr<arrow::RecordBatchReader> projection_reader =
      std::make_shared<ProjectionReader>(projection_schema, reader, options);
  return projection_reader;
}

std::shared_ptr<arrow::Schema> ProjectionReader::schema() const {
  // TODO
  return nullptr;
}

arrow::Status ProjectionReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  std::shared_ptr<arrow::RecordBatch> tmp;
  auto s = reader_->ReadNext(&tmp);
  if (!s.ok()) {
    return s;
  }

  if (!tmp) {
    batch = nullptr;
    return arrow::Status::OK();
  }

  std::vector<std::shared_ptr<arrow::Array>> projection_cols;
  for (int i = 0; i < tmp->num_columns(); ++i) {
    auto col_name = tmp->column_name(i);
    if (std::find(options_->columns.begin(), options_->columns.end(), col_name) != options_->columns.end()) {
      projection_cols.push_back(tmp->column(i));
    }
  }

  *batch = arrow::RecordBatch::Make(schema_, tmp->num_rows(), projection_cols);
  return arrow::Status::OK();
}
}  // namespace milvus_storage
