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
#include "milvus-storage/packed/column_group_writer.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

struct MemoryComparator {
  bool operator()(const std::pair<GroupId, size_t>& a, const std::pair<GroupId, size_t>& b) const {
    return a.second < b.second;
  }
};

using MemoryMaxHeap =
    std::priority_queue<std::pair<GroupId, size_t>, std::vector<std::pair<GroupId, size_t>>, MemoryComparator>;

class PackedRecordBatchWriter {
  public:
  PackedRecordBatchWriter(size_t memory_limit,
                          std::shared_ptr<arrow::Schema> schema,
                          std::shared_ptr<arrow::fs::FileSystem> fs,
                          const std::string& file_path,
                          const StorageConfig& storage_config);

  // Put the record batch into the corresponding column group,
  // , and write the maximum buffer of column group to the file.
  Status Write(const std::shared_ptr<arrow::RecordBatch>& record);
  // Close the writer and write the mapping of column offset to the metadata of parquet file.
  Status Close();

  private:
  // split first buffer into column groups based on column size
  // and init column group writer and put column groups into max heap
  Status splitAndWriteFirstBuffer();

  Status writeWithSplitIndex(const std::shared_ptr<arrow::RecordBatch>& record, size_t batch_size);
  Status balanceMaxHeap();
  Status flushRemainingBuffer();
  Status flushUnsplittedBuffer();

  std::vector<std::shared_ptr<arrow::RecordBatch>> buffered_batches_;
  bool size_split_done_;
  size_t memory_limit_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  const std::string file_path_;
  const StorageConfig storage_config_;
  size_t current_memory_usage_;
  std::vector<std::unique_ptr<ColumnGroupWriter>> group_writers_;
  std::vector<std::vector<int>> group_indices_;
  IndicesBasedSplitter splitter_;
  MemoryMaxHeap max_heap_;
};

class ColumnGroupSplitter {
  public:
  ColumnGroupSplitter();
  void splitColumns(const std::vector<std::vector<std::string>>& columns);

  private:
  std::vector<std::shared_ptr<ColumnGroupWriter>> columnGroupWriters_;
};

}  // namespace milvus_storage
