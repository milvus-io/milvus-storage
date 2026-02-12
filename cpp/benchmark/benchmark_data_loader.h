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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>

#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage {
namespace benchmark {

//=============================================================================
// Data Loader Type
//=============================================================================

enum class DataLoaderType {
  SYNTHETIC = 0,  // Use CreateTestData/CreateTestSchema
  MILVUS_SEGMENT  // Load from Milvus segment directory
};

//=============================================================================
// BenchmarkDataLoader - Abstract interface for loading benchmark data
//=============================================================================

class BenchmarkDataLoader {
  public:
  virtual ~BenchmarkDataLoader() = default;

  // Load data from source
  virtual arrow::Status Load() = 0;

  // Get merged schema (all columns)
  virtual std::shared_ptr<arrow::Schema> GetSchema() const = 0;

  // Get data as RecordBatchReader (for streaming reads)
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetRecordBatchReader() const = 0;

  // Get data as Table (loads all data into memory - use with caution for large datasets)
  virtual std::shared_ptr<arrow::Table> GetTable() const = 0;

  // Get a single RecordBatch (convenience method, loads all data)
  // For large datasets, prefer GetRecordBatchReader()
  virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordBatch() const = 0;

  // Get schema-based patterns string for Writer policy
  // Format: "col1,col2;col3,col4" where each group is separated by semicolon
  virtual std::string GetSchemaBasePatterns() const = 0;

  // Get number of rows
  virtual int64_t NumRows() const = 0;

  // Get total data size in bytes (approximate)
  virtual int64_t GetDataSize() const = 0;

  // Get projection columns (scalar columns only)
  virtual std::shared_ptr<std::vector<std::string>> GetScalarProjection() const = 0;

  // Get projection columns (vector columns only)
  virtual std::shared_ptr<std::vector<std::string>> GetVectorProjection() const = 0;

  // Get data source description (for logging/labels)
  virtual std::string GetDescription() const = 0;
};

//=============================================================================
// SyntheticDataLoader - Generate synthetic test data
//=============================================================================

struct SyntheticDataConfig {
  size_t num_rows = 40960;
  size_t vector_dim = 128;
  size_t string_length = 128;
  bool random_data = true;

  static SyntheticDataConfig Small() { return {4096, 128, 128, true}; }
  static SyntheticDataConfig Medium() { return {40960, 128, 128, true}; }
  static SyntheticDataConfig Large() { return {409600, 128, 128, true}; }
};

class SyntheticDataLoader : public BenchmarkDataLoader {
  public:
  explicit SyntheticDataLoader(const SyntheticDataConfig& config = SyntheticDataConfig::Medium());

  arrow::Status Load() override;

  std::shared_ptr<arrow::Schema> GetSchema() const override { return schema_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetRecordBatchReader() const override;

  std::shared_ptr<arrow::Table> GetTable() const override { return table_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordBatch() const override;

  std::string GetSchemaBasePatterns() const override;

  int64_t NumRows() const override { return table_ ? table_->num_rows() : 0; }

  int64_t GetDataSize() const override;

  std::shared_ptr<std::vector<std::string>> GetScalarProjection() const override;

  std::shared_ptr<std::vector<std::string>> GetVectorProjection() const override;

  std::string GetDescription() const override;

  private:
  SyntheticDataConfig config_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Table> table_;
};

//=============================================================================
// MilvusSegmentLoader - Load data from Milvus segment directory
//=============================================================================

class MilvusSegmentLoader : public BenchmarkDataLoader {
  public:
  struct ColumnGroupInfo {
    int64_t group_id;
    std::string file_path;
    std::shared_ptr<arrow::Schema> schema;
    std::shared_ptr<arrow::Table> table;
  };

  explicit MilvusSegmentLoader(const std::string& segment_path);

  arrow::Status Load() override;

  std::shared_ptr<arrow::Schema> GetSchema() const override { return merged_schema_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetRecordBatchReader() const override;

  std::shared_ptr<arrow::Table> GetTable() const override { return merged_table_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordBatch() const override;

  std::string GetSchemaBasePatterns() const override;

  int64_t NumRows() const override { return merged_table_ ? merged_table_->num_rows() : 0; }

  int64_t GetDataSize() const override;

  std::shared_ptr<std::vector<std::string>> GetScalarProjection() const override;

  std::shared_ptr<std::vector<std::string>> GetVectorProjection() const override;

  std::string GetDescription() const override;

  // Get column group info (for advanced use)
  const std::map<int64_t, ColumnGroupInfo>& GetColumnGroups() const { return column_groups_; }

  private:
  arrow::Status LoadColumnGroup(int64_t group_id, const std::string& file_path);
  arrow::Status BuildMergedData();

  std::string segment_path_;
  std::map<int64_t, ColumnGroupInfo> column_groups_;
  std::shared_ptr<arrow::Schema> merged_schema_;
  std::shared_ptr<arrow::Table> merged_table_;
};

//=============================================================================
// Factory function to create data loader
//=============================================================================

// Create data loader based on type
// For SYNTHETIC: config should be SyntheticDataConfig (or use default)
// For MILVUS_SEGMENT: path should be segment directory path
std::unique_ptr<BenchmarkDataLoader> CreateDataLoader(
    DataLoaderType type,
    const std::string& path = "",
    const SyntheticDataConfig& config = SyntheticDataConfig::Medium());

// Create data loader from environment variable
// If CUSTOM_SEGMENT_PATH is set, use MilvusSegmentLoader
// Otherwise, use SyntheticDataLoader with specified config
std::unique_ptr<BenchmarkDataLoader> CreateDataLoaderFromEnv(
    const SyntheticDataConfig& fallback_config = SyntheticDataConfig::Medium());

}  // namespace benchmark
}  // namespace milvus_storage
