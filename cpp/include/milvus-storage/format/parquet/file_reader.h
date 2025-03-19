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
#include "milvus-storage/format/reader.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/storage/options.h"
#include "milvus-storage/common/config.h"
namespace milvus_storage {

class FileRecordBatchReader : public arrow::RecordBatchReader {
  public:
  /**
   * @brief FileRecordBatchReader reads num of row groups starting from row_group_offset with memory constraints.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param buffer_size Memory limit for reading row groups.
   */
  FileRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                        const std::string& path,
                        const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE);

  arrow::Status SetRowGroupOffsetAndCount(size_t row_group_offset, size_t row_group_num);

  std::shared_ptr<arrow::Schema> schema() const;
  /**
   * @brief Returns each row group size of row groups in the Parquet file.
   */
  std::vector<size_t> GetRowGroupSizes();

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
  std::unique_ptr<parquet::arrow::FileReader> file_reader_;
  std::shared_ptr<parquet::FileMetaData> metadata_;
  size_t current_row_group_ = 0;
  size_t row_group_end_ = 0;
  size_t read_count_ = 0;

  int64_t buffer_size_;
  std::vector<size_t> row_group_sizes_;
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
