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
#include <utility>

#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"

namespace milvus_storage::parquet {

struct RowGroupIndex {
  size_t file_index;
  size_t row_group_index_in_file;
  size_t row_index_in_file;
  size_t size;
  size_t row_group_index;
  size_t row_index;
};

class ParquetChunkReader : public internal::api::ColumnGroupReader {
  public:
  /**
   * @brief FileRowGroupReader reads specified row groups. The schema is the same as the file schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param schema The Arrow schema.
   * @param paths Paths to the Parquet files.
   * @param row_nums Row numbers in each file.
   * @param reader_props The reader properties.
   * @param needed_columns Subset of columns to read (empty = all columns)
   */
  ParquetChunkReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     std::vector<std::string> paths,
                     ::parquet::ReaderProperties reader_props,
                     std::vector<std::string> needed_columns)
      : fs_(std::move(fs)),
        schema_(nullptr),
        paths_(std::move(paths)),
        reader_props_(std::move(reader_props)),
        needed_columns_(std::move(needed_columns)) {}

  [[nodiscard]] arrow::Status open() override;

  [[nodiscard]] size_t total_number_of_chunks() const override;

  [[nodiscard]] size_t total_rows() const override;

  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(
      const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_rows(int64_t chunk_index) override;

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> paths_;
  ::parquet::ReaderProperties reader_props_;
  std::vector<std::string> needed_columns_;
  size_t total_rows_;

  std::vector<int> needed_column_indices_;
  std::vector<std::shared_ptr<::parquet::arrow::FileReader>> file_readers_;
  std::vector<std::shared_ptr<PackedFileMetadata>> file_metadatas_;
  std::vector<RowGroupIndex> row_group_indices_;
};

}  // namespace milvus_storage::parquet
