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

#include "test_env.h"

#include <cassert>
#include <unistd.h>
#include <algorithm>
#include <random>

#include <arrow/type_fwd.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage {

bool IsCloudEnv() {
  auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
  return storage_type == "remote";
}

arrow::Status InitTestProperties(api::Properties& properties, std::string address, std::string root_path) {
  auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");

  if (storage_type == "local" || storage_type.empty()) {
    api::SetValue(properties, PROPERTY_FS_ADDRESS, address.c_str());
    api::SetValue(properties, PROPERTY_FS_ROOT_PATH, root_path.c_str());
  } else if (storage_type == "remote") {
    api::SetValue(properties, PROPERTY_FS_CLOUD_PROVIDER, GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("aws").c_str());
    api::SetValue(properties, PROPERTY_FS_ADDRESS, GetEnvVar(ENV_VAR_ADDRESS).ValueOr("http://localhost:9000").c_str());
    api::SetValue(properties, PROPERTY_FS_BUCKET_NAME, GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("test-bucket").c_str());
    api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_ID,
                  GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("minioadmin").c_str());
    api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_VALUE,
                  GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("minioadmin").c_str());
    api::SetValue(properties, PROPERTY_FS_REGION, GetEnvVar(ENV_VAR_REGION).ValueOr("").c_str());
    api::SetValue(properties, PROPERTY_FS_ROOT_PATH, GetEnvVar(ENV_VAR_ROOT_PATH).ValueOr("/").c_str());
  } else {
    return arrow::Status::Invalid("Unknown STORAGE_TYPE: " + storage_type);
  }

  api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, storage_type.c_str());
  return arrow::Status::OK();
}

arrow::Result<milvus_storage::ArrowFileSystemConfig> GetFileSystemConfig(const api::Properties& properties) {
  milvus_storage::ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
  return fs_config;
}

arrow::Result<milvus_storage::ArrowFileSystemPtr> GetFileSystem(const api::Properties& properties) {
  milvus_storage::ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
  ARROW_ASSIGN_OR_RAISE(milvus_storage::ArrowFileSystemPtr fs, milvus_storage::CreateArrowFileSystem(fs_config));
  return fs;
}

std::string GetTestBasePath(std::string dir) {
  std::string base_path;
  if (!IsCloudEnv()) {
    base_path = dir;
    return base_path;
  }
  auto bucket_name = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("test-bucket");
  base_path = bucket_name + "/" + dir;
  return base_path;
}

arrow::Status CreateTestDir(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& path) {
  assert(fs != nullptr);
  return fs->CreateDir(path);
}

arrow::Status DeleteTestDir(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& path, bool allow_missing) {
  assert(fs != nullptr);
  return fs->DeleteDirContents(path, allow_missing);
}

arrow::Result<std::shared_ptr<arrow::Schema>> CreateTestSchema(std::array<bool, 4> needed_columns) {
  std::vector<std::shared_ptr<arrow::Field>> fields;

  if (needed_columns[0]) {
    fields.emplace_back(
        arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})));
  }
  if (needed_columns[1]) {
    fields.emplace_back(
        arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"101"})));
  }
  if (needed_columns[2]) {
    fields.emplace_back(
        arrow::field("value", arrow::float64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"102"})));
  }
  if (needed_columns[3]) {
    fields.emplace_back(arrow::field("vector", arrow::list(arrow::float32()), false,
                                     arrow::key_value_metadata({"PARQUET:field_id"}, {"103"})));
  }

  if (fields.empty()) {
    return arrow::Status::Invalid("At least one column must be needed");
  }

  return arrow::schema(fields);
}

static std::string generateRandomString(int max_len) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> lengthDist(1, max_len);
  std::uniform_int_distribution<> charDist(33, 126);

  int length = lengthDist(gen);
  std::string result;
  for (int i = 0; i < length; ++i) {
    result += static_cast<char>(charDist(gen));
  }

  return result;
}

template <typename T>
static T generateRandomInt() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  return dist(gen);
}

template <typename T>
static T generateRandomReal() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<T> dist(std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::max());
  return dist(gen);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateTestData(std::shared_ptr<arrow::Schema> schema,
                                                                  bool randdata,
                                                                  size_t num_rows,
                                                                  size_t vector_dim,
                                                                  size_t str_length,
                                                                  std::array<bool, 4> needed_columns) {
  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  arrow::DoubleBuilder value_builder;
  arrow::ListBuilder vector_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::shared_ptr<arrow::Array> id_array, name_array, value_array, vector_array;

  srand(static_cast<unsigned>(time(0)));
  for (int64_t i = 0; i < num_rows; ++i) {
    if (needed_columns[0]) {
      ARROW_RETURN_NOT_OK(randdata ? id_builder.Append(generateRandomInt<int64_t>()) : id_builder.Append(i));
    }

    if (needed_columns[1]) {
      ARROW_RETURN_NOT_OK(
          name_builder.Append(randdata ? generateRandomString(str_length) : "name_" + std::to_string(i)));
    }

    if (needed_columns[2]) {
      ARROW_RETURN_NOT_OK(value_builder.Append(randdata ? generateRandomReal<double>() : i * 1.5));
    }

    if (needed_columns[3]) {
      // Create vector data
      auto vector_element_builder = static_cast<arrow::FloatBuilder*>(vector_builder.value_builder());
      ARROW_RETURN_NOT_OK(vector_builder.Append());
      for (int j = 0; j < vector_dim; ++j) {
        // Introduce some nulls randomly
        ARROW_RETURN_NOT_OK(randdata ? vector_element_builder->Append(static_cast<float>(rand()) /
                                                                      (static_cast<float>(RAND_MAX / 1000.0)))
                                     : vector_element_builder->Append(i * 0.1f + j));
      }
    }
  }

  if (needed_columns[0]) {
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
    arrays.emplace_back(id_array);
  }

  if (needed_columns[1]) {
    ARROW_RETURN_NOT_OK(name_builder.Finish(&name_array));
    arrays.emplace_back(name_array);
  }

  if (needed_columns[2]) {
    ARROW_RETURN_NOT_OK(value_builder.Finish(&value_array));
    arrays.emplace_back(value_array);
  }

  if (needed_columns[3]) {
    ARROW_RETURN_NOT_OK(vector_builder.Finish(&vector_array));
    arrays.emplace_back(vector_array);
  }

  return arrow::RecordBatch::Make(schema, num_rows, arrays);
}

arrow::Status ValidateRowAlignment(const std::shared_ptr<arrow::RecordBatch>& batch) {
  // Validate that data is properly aligned across columns
  // This checks that for each row, the data follows the expected pattern
  std::shared_ptr<arrow::StringArray> name_str_column = nullptr;
  std::shared_ptr<arrow::StringViewArray> name_strview_column = nullptr;

  // Get columns
  auto id_column = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
  if (batch->column(1)->type()->id() == arrow::Type::STRING) {
    name_str_column = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
  } else if (batch->column(1)->type()->id() == arrow::Type::STRING_VIEW) {
    name_strview_column = std::static_pointer_cast<arrow::StringViewArray>(batch->column(1));
  } else {
    return arrow::Status::Invalid("Column 1 is not of type STRING or STRING_VIEW");
  }

  auto value_column = std::static_pointer_cast<arrow::DoubleArray>(batch->column(2));
  auto vector_column = std::static_pointer_cast<arrow::ListArray>(batch->column(3));

  for (int64_t row = 0; row < batch->num_rows(); ++row) {
    // Skip null id rows
    if (id_column->IsNull(row)) {
      continue;
    }

    int64_t id_value = id_column->Value(row);
    int64_t original_id = id_value % 100;  // Original ID in test data (0-99)

    // Verify name matches expected pattern
    if (name_str_column && !name_str_column->IsNull(row)) {
      std::string name_value = name_str_column->GetString(row);
      std::string expected_name = "name_" + std::to_string(original_id);
      if (name_value != expected_name) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": name mismatch for id " +
                                      std::to_string(id_value));
      }
    }

    if (name_strview_column && !name_strview_column->IsNull(row)) {
      std::string name_value = name_strview_column->GetString(row);
      std::string expected_name = "name_" + std::to_string(original_id);
      if (name_value != expected_name) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": name mismatch for id " +
                                      std::to_string(id_value));
      }
    }

    // Verify value matches expected pattern
    if (!value_column->IsNull(row)) {
      double value_val = value_column->Value(row);
      double expected_value = original_id * 1.5;
      if (value_val != expected_value) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": value mismatch for id " +
                                      std::to_string(id_value));
      }
    }

    // Verify vector has expected structure (4 elements)
    if (!vector_column->IsNull(row)) {
      auto vector_slice = vector_column->value_slice(row);
      auto float_array = std::static_pointer_cast<arrow::FloatArray>(vector_slice);
      if (!float_array) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": vector column is not a FloatArray");
      }
      if (float_array->length() != 4) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": vector length mismatch for id " +
                                      std::to_string(id_value));
      }

      // Check vector values
      for (int j = 0; j < 4; ++j) {
        if (!float_array->IsNull(j)) {
          float expected_vector_value = original_id * 0.1f + j;
          if (float_array->Value(j) != expected_vector_value) {
            return arrow::Status::Invalid("Row " + std::to_string(row) + ", vector[" + std::to_string(j) +
                                          "]: value mismatch for id " + std::to_string(id_value));
          }
        }
      }
    }
  }

  return arrow::Status::OK();
}

}  // namespace milvus_storage