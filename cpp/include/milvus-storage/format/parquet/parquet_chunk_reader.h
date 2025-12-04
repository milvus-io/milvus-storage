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
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::parquet {
using internal::api::ChunkInfo;
using internal::api::RowGroupInfo;
class ParquetChunkReader : public internal::api::ColumnGroupReaderInternal {
  public:
  /**
   * @brief FileRowGroupReader reads specified row groups. The schema is the same as the file schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param paths Paths to the Parquet files.
   * @param reader_props The reader properties.
   * @param needed_columns Subset of columns to read (empty = all columns)
   */
  ParquetChunkReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                     const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                     const milvus_storage::api::Properties& properties,
                     const std::vector<std::string>& needed_columns,
                     const std::function<std::string(const std::string&)>& key_retriever);

  [[nodiscard]] arrow::Result<std::pair<std::vector<ChunkInfo>, std::vector<std::vector<RowGroupInfo>>>> open()
      override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunk(size_t file_index,
                                                                       const std::vector<RowGroupInfo>& row_group_info,
                                                                       const int& rg_index_in_file) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunks(
      size_t file_index,
      const std::vector<RowGroupInfo>& row_group_info,
      const std::vector<int>& rg_indices_in_file) override;

  protected:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  std::vector<milvus_storage::api::ColumnGroupFile> cg_files_;
  milvus_storage::api::Properties properties_;
  ::parquet::ReaderProperties reader_props_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  std::vector<int> needed_column_indices_;
  std::vector<std::shared_ptr<::parquet::arrow::FileReader>> file_readers_;
};

}  // namespace milvus_storage::parquet
