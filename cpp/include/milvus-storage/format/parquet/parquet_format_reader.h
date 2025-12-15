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

#pragma once

#include <memory>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::parquet {

class ParquetFormatReader final : public FormatReader {
  public:
  ParquetFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                      const std::string& path,
                      const milvus_storage::api::Properties& properties,
                      const std::vector<std::string>& needed_columns,
                      const std::function<std::string(const std::string&)>& key_retriever);

  // open the file
  [[nodiscard]] arrow::Status open() override;

  // get the row group infos
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;

  // get the chunk
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;

  // get the chunks
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;

  // get the chunk indices
  [[nodiscard]] arrow::Result<std::vector<int>> get_chunk_indices(const std::vector<int64_t>& row_indices);

  // take the rows
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;

  // read with range
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;

  private:
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunks_internal(
      const std::vector<int>& rg_indices_in_file);

  ParquetFormatReader(const ParquetFormatReader& other, std::unique_ptr<::parquet::arrow::FileReader> file_reader);

  std::string path_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  // init after open()
  std::vector<int> needed_column_indices_;
  std::vector<RowGroupInfo> row_group_infos_;
  std::unique_ptr<::parquet::arrow::FileReader> file_reader_;
};  // ParquetFormatReader

}  // namespace milvus_storage::parquet
