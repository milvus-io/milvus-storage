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
#include "format/reader.h"
#include "parquet/arrow/reader.h"
#include "storage/options.h"
#include "common/config.h"
namespace milvus_storage {

class FileRecordBatchReader : public arrow::RecordBatchReader {
  public:
  /**
   * @brief FileRecordBatchReader reads num of row groups starting from row_group_offset with memory constraints.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param schema Expected schema of the Parquet file.
   * @param buffer_size Memory limit for reading row groups.
   * @param row_group_offset The starting row group index to read.
   * @param row_group_num The number of row groups to read.
   */
  FileRecordBatchReader(arrow::fs::FileSystem& fs,
                        const std::string& path,
                        const std::shared_ptr<arrow::Schema>& schema,
                        const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
                        const size_t row_group_offset = 0,
                        const size_t row_group_num = std::numeric_limits<size_t>::max());

  /**
   * @brief Returns the schema of the Parquet file.
   *
   * @return A shared pointer to the Arrow schema.
   */
  std::shared_ptr<arrow::Schema> schema() const;

  /**
   * @brief Reads the next record batch from the file.
   *
   * @param out A shared pointer to the output record batch.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out);

  /**
   * @brief Closes the reader and releases resources.
   *
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status Close();

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::unique_ptr<parquet::arrow::FileReader> file_reader_;
  size_t current_row_group_ = 0;
  size_t read_count_ = 0;

  int64_t buffer_size_;
  std::vector<size_t> row_group_sizes_;
  size_t row_group_offset_;
};

class ParquetFileReader : public Reader {
  public:
  ParquetFileReader(std::unique_ptr<parquet::arrow::FileReader> reader);

  void Close() override {}

  Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) override;

  private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
};
}  // namespace milvus_storage
