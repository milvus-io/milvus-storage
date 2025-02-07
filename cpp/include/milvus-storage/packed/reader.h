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

#include "milvus-storage/packed/chunk_manager.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/common/config.h"
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
   * @brief Open a packed reader to read needed columns in the specified path.
   *
   * @param fs Arrow file system.
   * @param path The root path of the packed files to read.
   * @param origin_schema The original schema of data.
   * @param needed_columns The needed columns to read from the original schema.
   * @param buffer_size The max buffer size of the packed reader.
   */
  PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                          const std::string& file_path,
                          const std::shared_ptr<arrow::Schema> origin_schema,
                          const std::set<int>& needed_columns,
                          const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE);

  /**
   * @brief Return the schema of needed columns.
   */
  std::shared_ptr<arrow::Schema> schema() const override;

  /**
   * @brief Read next batch of arrow record batch to the specifed pointer.
   *        If the data is drained, return nullptr.
   *
   * @param batch The record batch pointer specified to read.
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  /**
   * @brief Close the reader and clean up resources.
   */
  arrow::Status Close() override;

  private:
  void init(arrow::fs::FileSystem& fs,
            const std::string& file_path,
            const std::shared_ptr<arrow::Schema> origin_schema,
            const std::set<int>& needed_columns,
            const int64_t buffer_size);

  Status initNeededSchema(const std::set<int>& needed_columns, const std::shared_ptr<arrow::Schema> origin_schema);

  Status initColumnOffsets(arrow::fs::FileSystem& fs, const std::set<int>& needed_columns, size_t num_fields);
  // Advance buffer to fill the expected buffer size
  arrow::Status advanceBuffer();

  std::vector<const arrow::Array*> collectChunks(int64_t chunksize) const;

  private:
  std::shared_ptr<arrow::Schema> needed_schema_;
  std::shared_ptr<arrow::Schema> origin_schema_;

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
  std::set<int> needed_paths_;
  std::vector<std::vector<size_t>> row_group_sizes_;
  const std::string file_path_;
  int read_count_;
};

}  // namespace milvus_storage
