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
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

class PackedRecordBatchWriter {
  private:
  struct MemoryComparator final {
    bool operator()(const std::pair<GroupId, size_t>& a, const std::pair<GroupId, size_t>& b) const {
      return a.second < b.second;
    }
  };

  using MemoryMaxHeap =
      std::priority_queue<std::pair<GroupId, size_t>, std::vector<std::pair<GroupId, size_t>>, MemoryComparator>;

  public:
  /**
   * @brief Open a packed writer to write needed columns into the specified paths.
   *
   * @param fs Arrow file system.
   * @param paths The paths to write, each path corresponds to a column group.
   * @param schema Arrow schema of written data.
   * @param storage_config Storage config.
   * @param column_groups The column groups to write in a file. Each column group is a vector of column indices.
   * @param buffer_size Max buffer size of the packed writer.
   * @param writer_props The writer properties.
   */
  PackedRecordBatchWriter(
      std::shared_ptr<arrow::fs::FileSystem> fs,
      std::vector<std::string>& paths,
      std::shared_ptr<arrow::Schema> schema,
      StorageConfig& storage_config,
      std::vector<std::vector<int>>& column_groups,
      size_t buffer_size = DEFAULT_WRITE_BUFFER_SIZE,
      std::shared_ptr<::parquet::WriterProperties> writer_props = ::parquet::default_writer_properties());

  /**
   * @brief Put the record batch into the corresponding column group,
   *        and write the maximum buffer of column group to a file in the specified path.
   *
   * @param record Arrow record batch to write.
   */
  Status Write(const std::shared_ptr<arrow::RecordBatch>& record);

  /**
   * @brief Close the writer and write the mapping of column offset to the metadata of parquet file.
   */
  Status Close();

  Status AddUserMetadata(const std::string& key, const std::string& value);

  Status AddMetadataBuilder(const std::string& key,
                            const std::function<std::unique_ptr<MetadataBuilder>()>& builder_factory);

  private:
  // split first buffer into column groups based on column size
  // and init column group writer and put column groups into max heap
  Status splitAndWriteFirstBuffer();

  Status balanceMaxHeap();
  Status flushRemainingBuffer();
  std::shared_ptr<arrow::Schema> getColumnGroupSchema(const std::shared_ptr<arrow::Schema>& schema,
                                                      const std::vector<int>& column_indices);

  GroupFieldIDList group_field_id_list_;
  size_t buffer_size_;
  const std::vector<std::vector<int>> group_indices_;
  size_t current_memory_usage_;
  bool closed_ = false;

  std::vector<std::shared_ptr<arrow::RecordBatch>> buffered_batches_;
  std::vector<std::unique_ptr<milvus_storage::parquet::ParquetFileWriter>> group_writers_;

  IndicesBasedSplitter splitter_;
  MemoryMaxHeap max_heap_;
  std::shared_ptr<::parquet::WriterProperties> writer_props_;
  std::vector<std::pair<std::string, std::string>> user_metadata_;
};

class ColumnGroupSplitter {
  public:
  ColumnGroupSplitter();
  void splitColumns(const std::vector<std::vector<std::string>>& columns);

  private:
  std::vector<std::shared_ptr<milvus_storage::parquet::ParquetFileWriter>> columnGroupWriters_;
};

}  // namespace milvus_storage
