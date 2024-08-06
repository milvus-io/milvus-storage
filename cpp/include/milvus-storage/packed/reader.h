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

#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "storage/options.h"

namespace milvus_storage {

// Default number of rows to read when using ::arrow::RecordBatchReader
static constexpr int64_t DefaultBatchSize = 1024;
static constexpr int64_t DefaultBufferSize = 16 * 1024 * 1024;

class PackedRecordBatchReader : public arrow::RecordBatchReader {
  public:
  PackedRecordBatchReader(arrow::fs::FileSystem& fs,
                          std::vector<std::string>& paths,
                          std::shared_ptr<arrow::Schema> schema,
                          std::vector<std::pair<int, int>>& column_offsets,
                          std::vector<int>& needed_columns,
                          int64_t buffer_size = DefaultBufferSize);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  protected:
  private:
  // Advance buffer to fill the expected buffer size
  arrow::Status advance_buffer();
  // Open file readers
  arrow::Status open();

  size_t buffer_size_;
  size_t buffer_available_;

  // Files
  arrow::fs::FileSystem& fs_;
  std::vector<std::string>& paths_;
  std::set<int> needed_path_indices_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::unique_ptr<parquet::arrow::FileReader>> file_readers_;
  std::vector<std::pair<int, int>> needed_column_offsets_;
  std::vector<int> needed_columns_;

  // Internal table states
  std::vector<std::shared_ptr<arrow::Table>> tables_;
  int64_t limit_;
  std::vector<int64_t> row_offsets_;
  std::vector<int> row_group_offsets_;
  std::vector<int64_t> table_memory_sizes_;

  // Internal chunking states
  std::vector<int> chunk_numbers_;
  std::vector<int64_t> chunk_offsets_;
  int64_t absolute_row_position_;
};

}  // namespace milvus_storage
