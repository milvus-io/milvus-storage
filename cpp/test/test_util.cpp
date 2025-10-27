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

arrow::Status PrepareSimpleParquetFile(std::shared_ptr<arrow::Schema> schema,
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
    return arrow::Status::Invalid(write_status.ToString());
  }
  auto close_status = w.Close();
  if (!close_status.ok()) {
    return arrow::Status::Invalid(close_status.ToString());
  }
  return arrow::Status::OK();
}

std::string GetEnvVar(const std::string& var_name) {
  const char* value = std::getenv(var_name.c_str());
  return value ? std::string(value) : std::string();
}

void InitTestProperties(api::Properties& properties, std::string address, std::string root_path) {
  if (GetEnvVar(ENV_VAR_STORAGE_TYPE) == "local" || GetEnvVar(ENV_VAR_STORAGE_TYPE).empty()) {
    api::SetValue(properties, PROPERTY_FS_ADDRESS, address.c_str());
    api::SetValue(properties, PROPERTY_FS_ROOT_PATH, root_path.c_str());

  } else {
    // must be remote
    assert(GetEnvVar(ENV_VAR_STORAGE_TYPE) == "remote");

    api::SetValue(properties, PROPERTY_FS_ADDRESS,
                  GetEnvVar(ENV_VAR_ADDRESS).empty() ? "http://localhost:9000" : GetEnvVar(ENV_VAR_ADDRESS).c_str());

    api::SetValue(properties, PROPERTY_FS_BUCKET_NAME,
                  GetEnvVar(ENV_VAR_BUCKET_NAME).empty() ? "test-bucket" : GetEnvVar(ENV_VAR_BUCKET_NAME).c_str());

    api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_ID,
                  GetEnvVar(ENV_VAR_ACCESS_KEY_ID).empty() ? "minioadmin" : GetEnvVar(ENV_VAR_ACCESS_KEY_ID).c_str());

    api::SetValue(
        properties, PROPERTY_FS_ACCESS_KEY_VALUE,
        GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).empty() ? "minioadmin" : GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).c_str());

    api::SetValue(properties, PROPERTY_FS_REGION,
                  GetEnvVar(ENV_VAR_REGION).empty() ? "" : GetEnvVar(ENV_VAR_REGION).c_str());

    api::SetValue(properties, PROPERTY_FS_REGION,
                  GetEnvVar(ENV_VAR_ROOT_PATH).empty() ? "files" : GetEnvVar(ENV_VAR_ROOT_PATH).c_str());
  }
}

}  // namespace milvus_storage
