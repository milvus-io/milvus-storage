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
#include <memory>
#include <random>
#include <iostream>
#include <utility>

#include <arrow/type_fwd.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include "milvus-storage/common/layout.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage {
using namespace milvus_storage::api;

bool IsCloudEnv() {
  auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
  return storage_type == "remote";
}

arrow::Status InitTestProperties(api::Properties& properties) {
  auto storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");

  if (storage_type == "local" || storage_type.empty()) {
    api::SetValue(properties, PROPERTY_FS_ROOT_PATH, "/tmp/milvus-storage-test");
  } else if (storage_type == "remote") {
    api::SetValue(properties, PROPERTY_FS_CLOUD_PROVIDER,
                  GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr(kCloudProviderAWS).c_str());
    api::SetValue(properties, PROPERTY_FS_ADDRESS, GetEnvVar(ENV_VAR_ADDRESS).ValueOr("http://localhost:9000").c_str());
    api::SetValue(properties, PROPERTY_FS_BUCKET_NAME, GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("test-bucket").c_str());
    api::SetValue(properties, PROPERTY_FS_REGION, GetEnvVar(ENV_VAR_REGION).ValueOr("").c_str());

    auto use_ssl = GetEnvVar(ENV_VAR_USE_SSL).ValueOr("");
    if (use_ssl == "true" || use_ssl == "1") {
      api::SetValue(properties, PROPERTY_FS_USE_SSL, "true");
    }

    auto use_iam = GetEnvVar(ENV_VAR_USE_IAM).ValueOr("");
    if (use_iam == "true" || use_iam == "1") {
      api::SetValue(properties, PROPERTY_FS_USE_IAM, "true");
      // Azure IAM still needs the storage account name via ACCESS_KEY_ID
      api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_ID, GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("").c_str());
    } else {
      api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_ID,
                    GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("minioadmin").c_str());
      api::SetValue(properties, PROPERTY_FS_ACCESS_KEY_VALUE,
                    GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("minioadmin").c_str());
    }
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
  // Use get_file_system which handles caching automatically
  return milvus_storage::FilesystemCache::getInstance().get(properties);
}

std::string GetTestBasePath(const std::string& dir) { return dir; }

arrow::Status MoveTestBasePath(const milvus_storage::ArrowFileSystemPtr& fs,
                               const std::string& old_dir,
                               const std::string& new_dir) {
  assert(fs != nullptr);
  if (IsLocalFileSystem(fs)) {
    return fs->Move(old_dir, new_dir);
  } else {
    arrow::fs::FileSelector selector;
    selector.base_dir = old_dir;
    selector.recursive = true;

    ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

    for (const auto& file_info : file_infos) {
      const std::string& origin_path = file_info.path();
      if (file_info.type() != arrow::fs::FileType::File) {
        continue;
      }
      auto pos = origin_path.find(old_dir);
      if (pos == std::string::npos) {
        return arrow::Status::Invalid("Invalid path: " + origin_path);
      }

      std::string new_path = origin_path;
      new_path.replace(pos, old_dir.length(), new_dir);
      ARROW_RETURN_NOT_OK(fs->Move(origin_path, new_path));
    }
    return arrow::Status::OK();
  }
}
arrow::Status CreateTestDir(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& path) {
  assert(fs != nullptr);
  ARROW_RETURN_NOT_OK(fs->CreateDir(path));
  ARROW_RETURN_NOT_OK(fs->CreateDir(get_manifest_path(path)));
  ARROW_RETURN_NOT_OK(fs->CreateDir(get_data_path(path)));
  return arrow::Status::OK();
}

arrow::Status DeleteTestDir(const milvus_storage::ArrowFileSystemPtr& fs, const std::string& path, bool allow_missing) {
  assert(fs != nullptr);
  return fs->DeleteDirContents(path, allow_missing);
}

arrow::Result<std::shared_ptr<arrow::Schema>> CreateTestSchema(std::array<bool, 4> needed_columns, size_t vector_dim) {
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
    if (vector_dim > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return arrow::Status::Invalid("vector_dim exceeds fixed-size list limit: " + std::to_string(vector_dim));
    }
    fields.emplace_back(arrow::field(
        "vector",
        arrow::fixed_size_list(arrow::field("item", arrow::float32(), false), static_cast<int32_t>(vector_dim)),
        false,
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

static arrow::Result<size_t> ResolveVectorDim(const std::shared_ptr<arrow::Schema>& schema,
                                              std::array<bool, 4> needed_columns,
                                              size_t fallback_vector_dim) {
  if (!needed_columns[3]) {
    return fallback_vector_dim;
  }
  auto vector_field = schema->GetFieldByName("vector");
  if (!vector_field) {
    return arrow::Status::Invalid("vector column is needed but schema has no vector field");
  }
  if (vector_field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto vector_type = std::static_pointer_cast<arrow::FixedSizeListType>(vector_field->type());
    if (vector_type->value_type()->id() != arrow::Type::FLOAT) {
      return arrow::Status::Invalid("vector fixed-size-list child type must be float32, got " +
                                    vector_type->value_type()->ToString());
    }
    return static_cast<size_t>(vector_type->list_size());
  }
  if (vector_field->type()->id() == arrow::Type::LIST) {
    return fallback_vector_dim;
  }
  return arrow::Status::Invalid("vector column has unexpected type " + vector_field->type()->ToString());
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateTestData(std::shared_ptr<arrow::Schema> schema,
                                                                  int64_t start_offset,
                                                                  bool randdata,
                                                                  size_t num_rows,
                                                                  size_t vector_dim,
                                                                  size_t str_length,
                                                                  std::array<bool, 4> needed_columns) {
  ARROW_ASSIGN_OR_RAISE(auto actual_vector_dim, ResolveVectorDim(schema, needed_columns, vector_dim));
  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  arrow::DoubleBuilder value_builder;
  auto vector_value_builder = std::make_shared<arrow::FloatBuilder>();
  std::unique_ptr<arrow::ListBuilder> vector_list_builder;
  std::unique_ptr<arrow::FixedSizeListBuilder> vector_fixed_size_list_builder;
  if (needed_columns[3]) {
    auto vector_field = schema->GetFieldByName("vector");
    if (vector_field->type()->id() == arrow::Type::LIST) {
      vector_list_builder = std::make_unique<arrow::ListBuilder>(arrow::default_memory_pool(), vector_value_builder);
    } else if (vector_field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
      if (actual_vector_dim > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return arrow::Status::Invalid("vector_dim exceeds fixed-size list limit: " + std::to_string(actual_vector_dim));
      }
      vector_fixed_size_list_builder = std::make_unique<arrow::FixedSizeListBuilder>(
          arrow::default_memory_pool(), vector_value_builder, static_cast<int32_t>(actual_vector_dim));
    } else {
      return arrow::Status::Invalid("vector column has unexpected type " + vector_field->type()->ToString());
    }
  }
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::shared_ptr<arrow::Array> id_array, name_array, value_array, vector_array;

  srand(static_cast<unsigned>(time(nullptr)));
  for (int64_t i = start_offset; i < start_offset + num_rows; ++i) {
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
      if (vector_fixed_size_list_builder) {
        ARROW_RETURN_NOT_OK(vector_fixed_size_list_builder->Append());
      } else {
        ARROW_RETURN_NOT_OK(vector_list_builder->Append());
      }
      for (size_t j = 0; j < actual_vector_dim; ++j) {
        // Introduce some nulls randomly
        ARROW_RETURN_NOT_OK(randdata ? vector_value_builder->Append(static_cast<float>(rand()) /
                                                                    (static_cast<float>(RAND_MAX / 1000.0)))
                                     : vector_value_builder->Append(i * 0.1f + j));
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
    if (vector_fixed_size_list_builder) {
      ARROW_RETURN_NOT_OK(vector_fixed_size_list_builder->Finish(&vector_array));
    } else {
      ARROW_RETURN_NOT_OK(vector_list_builder->Finish(&vector_array));
    }
    arrays.emplace_back(vector_array);
  }

  return arrow::RecordBatch::Make(std::move(schema), num_rows, arrays);
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
  auto vector_column = batch->column(3);

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

    // Verify vector has expected structure
    if (!vector_column->IsNull(row)) {
      std::shared_ptr<arrow::FloatArray> float_array;
      int64_t value_offset = 0;
      int32_t vector_length = 0;
      if (vector_column->type_id() == arrow::Type::LIST) {
        auto list_column = std::static_pointer_cast<arrow::ListArray>(vector_column);
        float_array = std::static_pointer_cast<arrow::FloatArray>(list_column->value_slice(row));
        vector_length = static_cast<int32_t>(float_array->length());
      } else if (vector_column->type_id() == arrow::Type::FIXED_SIZE_LIST) {
        auto list_column = std::static_pointer_cast<arrow::FixedSizeListArray>(vector_column);
        vector_length = list_column->value_length();
        float_array = std::static_pointer_cast<arrow::FloatArray>(list_column->values());
        value_offset = (list_column->offset() + row) * list_column->value_length();
      } else {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": vector column has unexpected type " +
                                      vector_column->type()->ToString());
      }
      if (!float_array) {
        return arrow::Status::Invalid("Row " + std::to_string(row) + ": vector column is not a FloatArray");
      }

      // Check vector values
      for (int j = 0; j < vector_length; ++j) {
        auto value_index = value_offset + j;
        if (!float_array->IsNull(value_index)) {
          float expected_vector_value = original_id * 0.1f + j;
          if (float_array->Value(value_index) != expected_vector_value) {
            return arrow::Status::Invalid("Row " + std::to_string(row) + ", vector[" + std::to_string(j) +
                                          "]: value mismatch for id " + std::to_string(id_value));
          }
        }
      }
    }
  }

  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<ColumnGroupPolicy>> CreateSinglePolicy(const std::string& format,
                                                                     const std::shared_ptr<arrow::Schema>& schema) {
  auto properties = milvus_storage::api::Properties{};
  SetValue(properties, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SINGLE);
  SetValue(properties, PROPERTY_WRITER_FORMAT, format.c_str());

  return ColumnGroupPolicy::create_column_group_policy(properties, schema);
}

arrow::Result<std::unique_ptr<ColumnGroupPolicy>> CreateSchemaBasePolicy(const std::string& patterns,
                                                                         const std::string& format,
                                                                         const std::shared_ptr<arrow::Schema>& schema) {
  auto properties = milvus_storage::api::Properties{};
  SetValue(properties, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SCHEMA_BASED);
  SetValue(properties, PROPERTY_WRITER_SCHEMA_BASE_PATTERNS, patterns.c_str());
  SetValue(properties, PROPERTY_WRITER_FORMAT, format.c_str());

  return ColumnGroupPolicy::create_column_group_policy(properties, schema);
}

arrow::Result<std::unique_ptr<ColumnGroupPolicy>> CreateSizeBasePolicy(int64_t max_avg_column_size,
                                                                       int64_t max_columns_in_group,
                                                                       const std::string& format,
                                                                       const std::shared_ptr<arrow::Schema>& schema) {
  auto properties = milvus_storage::api::Properties{};
  SetValue(properties, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SIZE_BASED);
  SetValue(properties, PROPERTY_WRITER_SIZE_BASE_MACS, std::to_string(max_avg_column_size).c_str());
  SetValue(properties, PROPERTY_WRITER_SIZE_BASE_MCIG, std::to_string(max_columns_in_group).c_str());
  SetValue(properties, PROPERTY_WRITER_FORMAT, format.c_str());

  return ColumnGroupPolicy::create_column_group_policy(properties, schema);
}

arrow::Result<std::vector<int64_t>> GenerateSortedUniqueArray(int rand_counts,
                                                              int max_value,
                                                              bool print_out,
                                                              bool useShuffle) {
  if (rand_counts > max_value) {
    return arrow::Status::Invalid("rand_counts > max_value");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<int64_t> result;
  if (useShuffle && rand_counts * 2 < (max_value - 1)) {
    std::unordered_set<int64_t> seen;
    std::uniform_int_distribution<> dis(0, max_value - 1);

    while (result.size() < rand_counts) {
      int64_t num = dis(gen);
      if (seen.insert(num).second) {
        result.push_back(num);
      }
    }

    std::sort(result.begin(), result.end());
  } else {
    std::vector<int64_t> all_numbers(max_value);
    std::iota(all_numbers.begin(), all_numbers.end(), 0);

    for (int i = 0; i < std::min(rand_counts, max_value); ++i) {
      std::uniform_int_distribution<> dis(i, max_value - 1);
      int j = dis(gen);
      std::swap(all_numbers[i], all_numbers[j]);
    }

    result = std::vector<int64_t>(all_numbers.begin(), all_numbers.begin() + rand_counts);
    std::sort(result.begin(), result.end());
  }

  if (print_out) {
    std::cout << "Random array: " << std::endl;
    for (long long i : result) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  return result;
}

}  // namespace milvus_storage
