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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>

#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include "test_env.h"

namespace milvus_storage::benchmark {

namespace {

arrow::Result<std::shared_ptr<arrow::Schema>> CreateStreamingSyntheticSchema(
    const StreamingSyntheticDataConfig& config) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  if (config.columns[0]) {
    fields.emplace_back(
        arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"100"})));
  }
  if (config.columns[1]) {
    fields.emplace_back(
        arrow::field("name", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"101"})));
  }
  if (config.columns[2]) {
    fields.emplace_back(
        arrow::field("value", arrow::float64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"102"})));
  }
  if (config.columns[3]) {
    if (config.vector_layout == VectorLayout::kFloatList) {
      fields.emplace_back(arrow::field("vector", arrow::list(arrow::float32()), false,
                                       arrow::key_value_metadata({"PARQUET:field_id"}, {"103"})));
    } else {
      if (config.vector_dim > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / sizeof(float)) {
        return arrow::Status::Invalid("vector byte width exceeds fixed-size binary limit");
      }
      fields.emplace_back(arrow::field("vector", arrow::fixed_size_binary(config.vector_dim * sizeof(float)), false,
                                       arrow::key_value_metadata({"PARQUET:field_id"}, {"103"})));
    }
  }
  if (fields.empty()) {
    return arrow::Status::Invalid("At least one streaming data column must be selected");
  }
  return arrow::schema(std::move(fields));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateStreamingSyntheticBatch(
    const std::shared_ptr<arrow::Schema>& schema,
    const StreamingSyntheticDataConfig& config,
    int64_t start_offset,
    size_t num_rows) {
  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  arrow::DoubleBuilder value_builder;
  arrow::ListBuilder list_vector_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
  std::unique_ptr<arrow::FixedSizeBinaryBuilder> binary_vector_builder;
  std::vector<uint8_t> binary_vector_data;
  if (config.columns[3] && config.vector_layout == VectorLayout::kFixedSizeBinary) {
    const auto vector_byte_width = config.vector_dim * sizeof(float);
    binary_vector_builder =
        std::make_unique<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(vector_byte_width));
    binary_vector_data.resize(vector_byte_width);
  }

  std::mt19937 generator(static_cast<uint32_t>(start_offset));
  std::uniform_real_distribution<float> float_distribution(0.0f, 1000.0f);
  std::uniform_real_distribution<double> double_distribution(0.0, 1000.0);
  auto* list_value_builder = static_cast<arrow::FloatBuilder*>(list_vector_builder.value_builder());

  for (size_t row = 0; row < num_rows; ++row) {
    const auto source_row = start_offset + static_cast<int64_t>(row);
    if (config.columns[0]) {
      ARROW_RETURN_NOT_OK(id_builder.Append(source_row));
    }
    if (config.columns[1]) {
      std::string name;
      if (config.random_data) {
        name.resize(std::max<size_t>(1, config.string_length));
        for (size_t i = 0; i < name.size(); ++i) {
          name[i] = static_cast<char>('a' + ((source_row + static_cast<int64_t>(i)) % 26));
        }
      } else {
        name = "name_" + std::to_string(source_row);
      }
      ARROW_RETURN_NOT_OK(name_builder.Append(name));
    }
    if (config.columns[2]) {
      ARROW_RETURN_NOT_OK(value_builder.Append(config.random_data ? double_distribution(generator) : source_row * 1.5));
    }
    if (config.columns[3]) {
      if (config.vector_layout == VectorLayout::kFloatList) {
        ARROW_RETURN_NOT_OK(list_vector_builder.Append());
      }
      for (size_t dimension = 0; dimension < config.vector_dim; ++dimension) {
        const auto value =
            config.random_data ? float_distribution(generator) : static_cast<float>(source_row * 0.1f + dimension);
        if (config.vector_layout == VectorLayout::kFloatList) {
          ARROW_RETURN_NOT_OK(list_value_builder->Append(value));
        } else {
          std::memcpy(binary_vector_data.data() + dimension * sizeof(float), &value, sizeof(value));
        }
      }
      if (config.vector_layout == VectorLayout::kFixedSizeBinary) {
        ARROW_RETURN_NOT_OK(binary_vector_builder->Append(binary_vector_data.data()));
      }
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::shared_ptr<arrow::Array> array;
  if (config.columns[0]) {
    ARROW_RETURN_NOT_OK(id_builder.Finish(&array));
    arrays.emplace_back(std::move(array));
  }
  if (config.columns[1]) {
    ARROW_RETURN_NOT_OK(name_builder.Finish(&array));
    arrays.emplace_back(std::move(array));
  }
  if (config.columns[2]) {
    ARROW_RETURN_NOT_OK(value_builder.Finish(&array));
    arrays.emplace_back(std::move(array));
  }
  if (config.columns[3]) {
    if (config.vector_layout == VectorLayout::kFloatList) {
      ARROW_RETURN_NOT_OK(list_vector_builder.Finish(&array));
    } else {
      ARROW_RETURN_NOT_OK(binary_vector_builder->Finish(&array));
    }
    arrays.emplace_back(std::move(array));
  }
  return arrow::RecordBatch::Make(schema, static_cast<int64_t>(num_rows), std::move(arrays));
}

class StreamingSyntheticBatchReader final : public arrow::RecordBatchReader {
  public:
  StreamingSyntheticBatchReader(std::shared_ptr<arrow::Schema> schema, StreamingSyntheticDataConfig config)
      : schema_(std::move(schema)), config_(std::move(config)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    if (rows_read_ >= config_.num_rows) {
      *out = nullptr;
      return arrow::Status::OK();
    }
    const auto rows = std::min(config_.batch_rows, config_.num_rows - rows_read_);
    ARROW_ASSIGN_OR_RAISE(*out,
                          CreateStreamingSyntheticBatch(schema_, config_, static_cast<int64_t>(rows_read_), rows));
    rows_read_ += rows;
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::Schema> schema_;
  StreamingSyntheticDataConfig config_;
  size_t rows_read_ = 0;
};

class StreamingSyntheticDataLoader final : public BenchmarkDataLoader {
  public:
  explicit StreamingSyntheticDataLoader(StreamingSyntheticDataConfig config) : config_(std::move(config)) {}

  arrow::Status Load() override {
    if (config_.batch_rows == 0) {
      return arrow::Status::Invalid("Streaming batch_rows must be positive");
    }
    ARROW_ASSIGN_OR_RAISE(schema_, CreateStreamingSyntheticSchema(config_));
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Schema> GetSchema() const override { return schema_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetRecordBatchReader() const override {
    if (!schema_) {
      return arrow::Status::Invalid("Data not loaded");
    }
    return std::make_shared<StreamingSyntheticBatchReader>(schema_, config_);
  }

  std::shared_ptr<arrow::Table> GetTable() const override { return nullptr; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordBatch() const override {
    return arrow::Status::NotImplemented("Streaming benchmark data is available through GetRecordBatchReader only");
  }

  std::string GetSchemaBasePatterns() const override {
    if (config_.columns[0] || config_.columns[1] || config_.columns[2]) {
      return config_.columns[3] ? "id,name,value;vector" : "id,name,value";
    }
    return "vector";
  }

  int64_t NumRows() const override { return static_cast<int64_t>(config_.num_rows); }

  int64_t GetDataSize() const override {
    size_t bytes_per_row = 0;
    if (config_.columns[0]) {
      bytes_per_row += sizeof(int64_t);
    }
    if (config_.columns[1]) {
      bytes_per_row += config_.string_length;
    }
    if (config_.columns[2]) {
      bytes_per_row += sizeof(double);
    }
    if (config_.columns[3]) {
      bytes_per_row += config_.vector_dim * sizeof(float);
    }
    return static_cast<int64_t>(config_.num_rows * bytes_per_row);
  }

  std::shared_ptr<std::vector<std::string>> GetScalarProjection() const override {
    auto projection = std::make_shared<std::vector<std::string>>();
    if (config_.columns[0]) {
      projection->emplace_back("id");
    }
    if (config_.columns[1]) {
      projection->emplace_back("name");
    }
    if (config_.columns[2]) {
      projection->emplace_back("value");
    }
    return projection;
  }

  std::shared_ptr<std::vector<std::string>> GetVectorProjection() const override {
    auto projection = std::make_shared<std::vector<std::string>>();
    if (config_.columns[3]) {
      projection->emplace_back("vector");
    }
    return projection;
  }

  std::string GetDescription() const override { return config_.label; }

  private:
  StreamingSyntheticDataConfig config_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace

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
  if (!table_) {
    return 0;
  }
  int64_t size = 0;
  for (int i = 0; i < table_->num_columns(); ++i) {
    for (const auto& chunk : table_->column(i)->chunks()) {
      for (const auto& buffer : chunk->data()->buffers) {
        if (buffer) {
          size += buffer->size();
        }
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
  if (!merged_table_) {
    return 0;
  }
  int64_t size = 0;
  for (int i = 0; i < merged_table_->num_columns(); ++i) {
    for (const auto& chunk : merged_table_->column(i)->chunks()) {
      for (const auto& buffer : chunk->data()->buffers) {
        if (buffer) {
          size += buffer->size();
        }
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

std::string_view ReaderBenchmarkDatasetName(ReaderBenchmarkDataset dataset) {
  switch (dataset) {
    case ReaderBenchmarkDataset::kSyntheticSmall:
      return "SyntheticSmall";
    case ReaderBenchmarkDataset::kSyntheticMedium:
      return "SyntheticMedium";
    case ReaderBenchmarkDataset::kSyntheticLarge:
      return "SyntheticLarge";
    case ReaderBenchmarkDataset::kScalarMedium:
      return "ScalarMedium";
    case ReaderBenchmarkDataset::kRandomVector64MiB:
      return "RandomVector64MiB";
    case ReaderBenchmarkDataset::kLowEntropyVector256MiB:
      return "LowEntropyVector256MiB";
    case ReaderBenchmarkDataset::kRandomVector2GiB:
      return "RandomVector2GiB";
  }
  return "unknown";
}

std::unique_ptr<BenchmarkDataLoader> CreateStreamingSyntheticDataLoader(StreamingSyntheticDataConfig config) {
  return std::make_unique<StreamingSyntheticDataLoader>(std::move(config));
}

arrow::Result<std::unique_ptr<BenchmarkDataLoader>> CreateReaderBenchmarkDataLoader(ReaderBenchmarkDataset dataset) {
  constexpr size_t kVectorDimension = 256;
  constexpr size_t kVectorBatchRows = 65536;
  switch (dataset) {
    case ReaderBenchmarkDataset::kSyntheticSmall:
      return CreateStreamingSyntheticDataLoader({4096,
                                                 128,
                                                 128,
                                                 4096,
                                                 true,
                                                 {true, true, true, true},
                                                 VectorLayout::kFloatList,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kSyntheticMedium:
      return CreateStreamingSyntheticDataLoader({40960,
                                                 128,
                                                 128,
                                                 40960,
                                                 true,
                                                 {true, true, true, true},
                                                 VectorLayout::kFloatList,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kSyntheticLarge:
      return CreateStreamingSyntheticDataLoader({409600,
                                                 128,
                                                 128,
                                                 65536,
                                                 true,
                                                 {true, true, true, true},
                                                 VectorLayout::kFloatList,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kScalarMedium:
      return CreateStreamingSyntheticDataLoader({40960,
                                                 0,
                                                 128,
                                                 40960,
                                                 true,
                                                 {true, true, true, false},
                                                 VectorLayout::kFloatList,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kRandomVector64MiB:
      return CreateStreamingSyntheticDataLoader({65536,
                                                 kVectorDimension,
                                                 0,
                                                 kVectorBatchRows,
                                                 true,
                                                 {false, false, false, true},
                                                 VectorLayout::kFixedSizeBinary,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kLowEntropyVector256MiB:
      return CreateStreamingSyntheticDataLoader({262144,
                                                 kVectorDimension,
                                                 0,
                                                 kVectorBatchRows,
                                                 false,
                                                 {false, false, false, true},
                                                 VectorLayout::kFixedSizeBinary,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
    case ReaderBenchmarkDataset::kRandomVector2GiB:
      return CreateStreamingSyntheticDataLoader({2097152,
                                                 kVectorDimension,
                                                 0,
                                                 kVectorBatchRows,
                                                 true,
                                                 {false, false, false, true},
                                                 VectorLayout::kFixedSizeBinary,
                                                 std::string(ReaderBenchmarkDatasetName(dataset))});
  }
  return arrow::Status::Invalid("Unknown Reader benchmark dataset");
}

}  // namespace milvus_storage::benchmark
