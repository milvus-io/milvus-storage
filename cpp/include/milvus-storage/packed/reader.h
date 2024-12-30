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

#include <packed/chunk_manager.h>
#include <packed/column_group.h>
#include "common/config.h"
#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <queue>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

struct RowOffsetComparator {
  bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const { return a.second > b.second; }
};

using RowOffsetMinHeap =
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, RowOffsetComparator>;

class PackedRecordBatchReader : public arrow::RecordBatchReader {
  public:
  /**
   * @brief PackedRecordBatchReader is responsible for reading and deserializing data from multiple Parquet files
   * into arrow RecordBatch under the given memory limit.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Parquet file paths to read.
   * @param schema Expected arrow schema reading from the Parquet files.
   * @param column_offsets tTe list of original column index and its path index.
   * @param needed_columns The set of columns needed to be read.
   * @param buffer_size Memory limit for reading.
   */
  PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                          const std::vector<std::string>& paths,
                          const std::shared_ptr<arrow::Schema> schema,
                          const std::vector<ColumnOffset>& column_offsets,
                          const std::set<int>& needed_columns,
                          const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE);

  /**
   * @brief Returns the schema of the RecordBatch being read.
   *
   * @return A shared pointer to the Arrow schema.
   */
  std::shared_ptr<arrow::Schema> schema() const override;

  /**
   * @brief Reads the next record batch from the file.
   *
   * @param out A shared pointer to the output record batch.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  /**
   * @brief Closes the reader and releases resources.
   *
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status Close() override;

  private:
  // Advance buffer to fill the expected buffer size
  arrow::Status advanceBuffer();
  std::vector<const arrow::Array*> collectChunks(int64_t chunksize) const;

  private:
  std::shared_ptr<arrow::Schema> schema_;

  size_t memory_limit_;
  size_t buffer_available_;
  std::vector<std::unique_ptr<parquet::arrow::FileReader>> file_readers_;
  std::vector<std::shared_ptr<arrow::KeyValueMetadata>> metadata_;
  std::vector<std::queue<std::shared_ptr<arrow::Table>>> tables_;
  std::vector<ColumnGroupState> column_group_states_;
  int64_t row_limit_;
  std::unique_ptr<ChunkManager> chunk_manager_;
  int64_t absolute_row_position_;
  std::vector<ColumnOffset> needed_column_offsets_;
  std::vector<std::vector<size_t>> row_group_sizes_;
  int read_count_;
};

}  // namespace milvus_storage
