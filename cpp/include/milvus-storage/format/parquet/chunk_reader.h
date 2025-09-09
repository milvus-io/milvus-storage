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
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/reader.h"

namespace milvus_storage::api {

class ParquetChunkReader : public ChunkReader {
  public:
  /**
   * @brief FileRowGroupReader reads specified row groups. The schema is the same as the file schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param reader_props The reader properties.
   * @param needed_columns Subset of columns to read (empty = all columns)
   */
  ParquetChunkReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     const std::string& path,
                     parquet::ReaderProperties reader_props = parquet::default_reader_properties(),
                     const std::vector<std::string>& needed_columns = {});

  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(
      const std::vector<int64_t>& row_indices) const override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) const override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunk_range(
      int64_t start_chunk_index, int64_t chunk_count) const override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) const override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_row_num(int64_t chunk_index) const override;

  std::shared_ptr<arrow::Schema> schema() const;

  protected:
  [[nodiscard]] arrow::Status validate_chunk_index(int64_t chunk_index) const override;

  private:
  Status init(std::shared_ptr<arrow::fs::FileSystem> fs,
              const std::string& path,
              parquet::ReaderProperties reader_props = parquet::default_reader_properties());

  std::vector<int> needed_column_indices_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<parquet::arrow::FileReader> file_reader_;

  std::shared_ptr<PackedFileMetadata> file_metadata_;
  std::vector<int64_t> num_rows_until_chunk_;
};

}  // namespace milvus_storage::api
