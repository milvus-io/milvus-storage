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
#include <unordered_set>
#include <vector>

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"

namespace milvus_storage::iceberg {

/// IcebergFormatReader wraps ParquetFormatReader and applies Iceberg
/// positional deletes on top. Delete metadata is stored as JSON in
/// ColumnGroupFile.metadata and parsed at open() time.
class IcebergFormatReader final : public FormatReader {
  public:
  IcebergFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                      const std::string& resolved_path,
                      const std::string& data_file_uri,
                      const std::vector<uint8_t>& delete_metadata,
                      const api::Properties& properties,
                      const std::vector<std::string>& needed_columns,
                      const std::function<std::string(const std::string&)>& key_retriever);

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  /// Parse delete metadata JSON and read positional delete files.
  [[nodiscard]] arrow::Status load_positional_deletes();

  /// Read a single positional delete Parquet file and collect positions
  /// that match the data file URI into deleted_positions_.
  [[nodiscard]] arrow::Status read_positional_delete_file(const std::string& delete_file_path);

  /// Filter a RecordBatch by removing rows at deleted positions.
  /// chunk_start is the global physical offset of the first row in the batch.
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> filter_batch(
      const std::shared_ptr<arrow::RecordBatch>& batch, size_t chunk_start);

  /// Constructor for clone_reader().
  IcebergFormatReader(std::shared_ptr<parquet::ParquetFormatReader> inner,
                      const std::string& data_file_uri,
                      const api::Properties& properties,
                      std::shared_ptr<std::unordered_set<int64_t>> deleted_positions);

  std::shared_ptr<parquet::ParquetFormatReader> inner_reader_;
  std::string data_file_uri_;
  std::vector<uint8_t> delete_metadata_;
  api::Properties properties_;

  // Populated during open(), shared across clones. Read-only after open().
  std::shared_ptr<std::unordered_set<int64_t>> deleted_positions_;
};

}  // namespace milvus_storage::iceberg
