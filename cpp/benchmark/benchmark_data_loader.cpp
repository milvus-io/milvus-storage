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

#include "benchmark_data_loader.h"

#include <filesystem>
#include <cstdlib>

#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include "test_env.h"

namespace milvus_storage {
namespace benchmark {

//=============================================================================
// SyntheticDataLoader Implementation
//=============================================================================

SyntheticDataLoader::SyntheticDataLoader(const SyntheticDataConfig& config) : config_(config) {}

arrow::Status SyntheticDataLoader::Load() {
  // Create schema using test helper
  ARROW_ASSIGN_OR_RAISE(schema_, CreateTestSchema());

  // Create test data
  ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema_, 0, config_.random_data, config_.num_rows,
                                                   config_.vector_dim, config_.string_length));

  // Convert to table
  table_ = arrow::Table::Make(schema_, batch->columns(), batch->num_rows());

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> SyntheticDataLoader::GetRecordBatchReader() const {
  if (!table_) {
    return arrow::Status::Invalid("Data not loaded");
  }
  return std::make_shared<arrow::TableBatchReader>(*table_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> SyntheticDataLoader::GetRecordBatch() const {
  if (!table_) {
    return arrow::Status::Invalid("Data not loaded");
  }
  arrow::TableBatchReader reader(*table_);
  reader.set_chunksize(table_->num_rows());
  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(reader.ReadNext(&batch));
  return batch;
}

std::string SyntheticDataLoader::GetSchemaBasePatterns() const {
  // Default synthetic schema: id, name, value, vector
  // Group 1: id, name, value (scalar)
  // Group 2: vector
  return "id,name,value;vector";
}

int64_t SyntheticDataLoader::GetDataSize() const {
  if (!table_)
    return 0;
  int64_t size = 0;
  for (int i = 0; i < table_->num_columns(); ++i) {
    for (const auto& chunk : table_->column(i)->chunks()) {
      for (const auto& buffer : chunk->data()->buffers) {
        if (buffer)
          size += buffer->size();
      }
    }
  }
  return size;
}

std::shared_ptr<std::vector<std::string>> SyntheticDataLoader::GetScalarProjection() const {
  auto projection = std::make_shared<std::vector<std::string>>();
  projection->push_back("id");
  projection->push_back("name");
  projection->push_back("value");
  return projection;
}

std::shared_ptr<std::vector<std::string>> SyntheticDataLoader::GetVectorProjection() const {
  auto projection = std::make_shared<std::vector<std::string>>();
  projection->push_back("vector");
  return projection;
}

std::string SyntheticDataLoader::GetDescription() const {
  return "synthetic/" + std::to_string(config_.num_rows) + "rows/" + std::to_string(config_.vector_dim) + "dim";
}

//=============================================================================
// MilvusSegmentLoader Implementation
//=============================================================================

MilvusSegmentLoader::MilvusSegmentLoader(const std::string& segment_path) : segment_path_(segment_path) {}

arrow::Status MilvusSegmentLoader::Load() {
  namespace fs = std::filesystem;

  if (!fs::exists(segment_path_)) {
    return arrow::Status::IOError("Segment path does not exist: " + segment_path_);
  }

  // Iterate through column group directories
  for (const auto& entry : fs::directory_iterator(segment_path_)) {
    if (!entry.is_directory()) {
      continue;
    }

    std::string dir_name = entry.path().filename().string();
    int64_t group_id;
    try {
      group_id = std::stoll(dir_name);
    } catch (...) {
      continue;  // Skip non-numeric directories
    }

    // Find parquet file in this column group directory
    for (const auto& file_entry : fs::directory_iterator(entry.path())) {
      if (file_entry.is_regular_file()) {
        std::string file_path = file_entry.path().string();
        ARROW_RETURN_NOT_OK(LoadColumnGroup(group_id, file_path));
        break;  // Only one file per column group
      }
    }
  }

  if (column_groups_.empty()) {
    return arrow::Status::Invalid("No column groups found in segment: " + segment_path_);
  }

  // Build merged schema and table
  ARROW_RETURN_NOT_OK(BuildMergedData());

  return arrow::Status::OK();
}

arrow::Status MilvusSegmentLoader::LoadColumnGroup(int64_t group_id, const std::string& file_path) {
  // Open parquet file
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(file_path));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  // Read schema
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));

  // Read table
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));

  column_groups_[group_id] = {group_id, file_path, schema, table};
  return arrow::Status::OK();
}

arrow::Status MilvusSegmentLoader::BuildMergedData() {
  // Collect all fields and columns
  std::vector<std::shared_ptr<arrow::Field>> all_fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> all_columns;

  for (const auto& [group_id, info] : column_groups_) {
    for (int i = 0; i < info.schema->num_fields(); ++i) {
      all_fields.push_back(info.schema->field(i));
      all_columns.push_back(info.table->column(i));
    }
  }

  merged_schema_ = arrow::schema(all_fields);
  merged_table_ = arrow::Table::Make(merged_schema_, all_columns);
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> MilvusSegmentLoader::GetRecordBatchReader() const {
  if (!merged_table_) {
    return arrow::Status::Invalid("Data not loaded");
  }
  return std::make_shared<arrow::TableBatchReader>(*merged_table_);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MilvusSegmentLoader::GetRecordBatch() const {
  if (!merged_table_) {
    return arrow::Status::Invalid("Data not loaded");
  }
  arrow::TableBatchReader reader(*merged_table_);
  reader.set_chunksize(merged_table_->num_rows());
  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_RETURN_NOT_OK(reader.ReadNext(&batch));
  return batch;
}

std::string MilvusSegmentLoader::GetSchemaBasePatterns() const {
  // Pattern format: "col1|col2,col3|col4" where:
  // - '|' separates columns within the same group
  // - ',' separates different groups
  std::string patterns;
  for (const auto& [group_id, info] : column_groups_) {
    if (!patterns.empty()) {
      patterns += ",";
    }
    for (int i = 0; i < info.schema->num_fields(); ++i) {
      if (i > 0) {
        patterns += "|";
      }
      patterns += info.schema->field(i)->name();
    }
  }
  return patterns;
}

int64_t MilvusSegmentLoader::GetDataSize() const {
  if (!merged_table_)
    return 0;
  int64_t size = 0;
  for (int i = 0; i < merged_table_->num_columns(); ++i) {
    for (const auto& chunk : merged_table_->column(i)->chunks()) {
      for (const auto& buffer : chunk->data()->buffers) {
        if (buffer)
          size += buffer->size();
      }
    }
  }
  return size;
}

std::shared_ptr<std::vector<std::string>> MilvusSegmentLoader::GetScalarProjection() const {
  auto projection = std::make_shared<std::vector<std::string>>();
  for (const auto& field : merged_schema_->fields()) {
    // Skip system columns
    if (field->name() == "RowID" || field->name() == "Timestamp") {
      continue;
    }
    // Skip vector columns (fixed_size_binary or list<float>)
    if (field->type()->id() == arrow::Type::FIXED_SIZE_BINARY || field->type()->id() == arrow::Type::LIST) {
      continue;
    }
    projection->push_back(field->name());
  }
  return projection;
}

std::shared_ptr<std::vector<std::string>> MilvusSegmentLoader::GetVectorProjection() const {
  auto projection = std::make_shared<std::vector<std::string>>();
  for (const auto& field : merged_schema_->fields()) {
    if (field->type()->id() == arrow::Type::FIXED_SIZE_BINARY || field->type()->id() == arrow::Type::LIST) {
      projection->push_back(field->name());
    }
  }
  return projection;
}

std::string MilvusSegmentLoader::GetDescription() const {
  namespace fs = std::filesystem;
  std::string segment_name = fs::path(segment_path_).filename().string();
  return "milvus/" + segment_name + "/" + std::to_string(NumRows()) + "rows";
}

//=============================================================================
// Factory Functions
//=============================================================================

std::unique_ptr<BenchmarkDataLoader> CreateDataLoader(DataLoaderType type,
                                                      const std::string& path,
                                                      const SyntheticDataConfig& config) {
  switch (type) {
    case DataLoaderType::SYNTHETIC:
      return std::make_unique<SyntheticDataLoader>(config);
    case DataLoaderType::MILVUS_SEGMENT:
      return std::make_unique<MilvusSegmentLoader>(path);
    default:
      return std::make_unique<SyntheticDataLoader>(config);
  }
}

std::unique_ptr<BenchmarkDataLoader> CreateDataLoaderFromEnv(const SyntheticDataConfig& fallback_config) {
  const char* segment_path = std::getenv("CUSTOM_SEGMENT_PATH");
  if (segment_path && segment_path[0] != '\0') {
    return std::make_unique<MilvusSegmentLoader>(segment_path);
  }
  return std::make_unique<SyntheticDataLoader>(fallback_config);
}

}  // namespace benchmark
}  // namespace milvus_storage
