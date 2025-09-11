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

#include "milvus-storage/common/metadata.h"
#include "milvus-storage/packed/chunk_manager.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/row_offset_heap.h"
#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

using RowOffsetMinHeap = milvus_storage::RowOffsetMinHeap;

class PackedRecordBatchReader : public arrow::RecordBatchReader {
  public:
  /**
   * @brief Open a packed reader to read needed columns in the specified path.
   *
   * @param fs Arrow file system.
   * @param paths Paths of the packed files to read.
   * @param schema The schema of data to read.
   * @param needed_columns The needed columns to read from the original schema.
   * @param buffer_size The max buffer size of the packed reader.
   * @param reader_props The reader properties.
   */
  PackedRecordBatchReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                          std::vector<std::string>& paths,
                          std::shared_ptr<arrow::Schema> schema,
                          int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
                          ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties());

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
   * @brief Returns packed file metadata for the i-th file.
   */
  std::shared_ptr<PackedFileMetadata> file_metadata(int i);

  /**
   * @brief Close the reader and clean up resources.
   */
  arrow::Status Close() override;

  private:
  Status init(std::shared_ptr<arrow::fs::FileSystem> fs,
              std::vector<std::string>& paths,
              std::shared_ptr<arrow::Schema> origin_schema,
              ::parquet::ReaderProperties& reader_props);

  Status schemaMatching(std::shared_ptr<arrow::fs::FileSystem> fs,
                        std::shared_ptr<arrow::Schema> schema,
                        std::vector<std::string>& paths,
                        ::parquet::ReaderProperties& reader_props);

  // Advance buffer to fill the expected buffer size
  arrow::Status advanceBuffer();

  int64_t get_next_row_group_size(int i);

  std::vector<const arrow::Array*> collectChunks(int64_t chunksize) const;

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> needed_schema_;
  FieldIDList field_id_list_;
  std::map<FieldID, ColumnOffset> field_id_mapping_;

  size_t memory_limit_;
  size_t memory_used_;
  std::vector<std::unique_ptr<::parquet::arrow::FileReader>> file_readers_;
  std::vector<std::queue<std::shared_ptr<arrow::Table>>> tables_;
  std::vector<ColumnGroupState> column_group_states_;
  int64_t row_limit_;
  std::unique_ptr<ChunkManager> chunk_manager_;
  int64_t absolute_row_position_;
  std::vector<ColumnOffset> needed_column_offsets_;
  std::set<std::string> needed_paths_;
  std::vector<std::shared_ptr<PackedFileMetadata>> metadata_list_;
  std::vector<int> file_reader_to_path_index_;
  int read_count_;
  size_t drained_files_;
};

}  // namespace milvus_storage
