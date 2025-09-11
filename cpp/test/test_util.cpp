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

#include "test_util.h"
#include <arrow/type_fwd.h>
#include "milvus-storage/format/parquet/file_writer.h"
#include "arrow/array/builder_primitive.h"
#include "milvus-storage/common/config.h"
namespace milvus_storage {
std::shared_ptr<arrow::Schema> CreateArrowSchema(std::vector<std::string> field_names,
                                                 std::vector<std::shared_ptr<arrow::DataType>> field_types) {
  arrow::FieldVector fields;
  for (int i = 0; i < field_names.size(); i++) {
    fields.push_back(arrow::field(field_names[i], field_types[i]));
  }
  return std::make_shared<arrow::Schema>(fields);
}

Status PrepareSimpleParquetFile(std::shared_ptr<arrow::Schema> schema,
                                std::shared_ptr<arrow::fs::FileSystem> fs,
                                const std::string& file_path,
                                int num_rows) {
  // TODO: parse schema and generate data
  auto conf = StorageConfig();
  milvus_storage::parquet::ParquetFileWriter w(schema, fs, file_path, conf);
  arrow::Int64Builder builder;
  for (int i = 0; i < num_rows; i++) {
    RETURN_ARROW_NOT_OK(builder.Append(i));
  }
  std::shared_ptr<arrow::Array> array;
  RETURN_ARROW_NOT_OK(builder.Finish(&array));
  auto batch = arrow::RecordBatch::Make(schema, num_rows, {array});
  auto write_status = w.Write(batch);
  if (!write_status.ok()) {
    return Status::ArrowError(write_status.ToString());
  }
  auto close_status = w.Close();
  if (!close_status.ok()) {
    return Status::ArrowError(close_status.ToString());
  }
  return Status::OK();
}
}  // namespace milvus_storage
