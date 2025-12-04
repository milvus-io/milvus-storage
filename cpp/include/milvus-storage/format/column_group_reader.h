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

#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/format/format.h"
#include "milvus-storage/properties.h"

namespace internal::api {

struct ChunkInfo {
  public:
  size_t file_index;               // current chunk belong which file
  size_t row_offset_in_row_group;  // the starting row offset of this row group in its file
  size_t row_offset_in_file;       // the starting row offset of file
  size_t number_of_rows;           // number of rows in this row group
  size_t row_group_index_in_file;  // the index of this row group in its file
  size_t global_row_end;           // the ending row offset of this row group in the whole chunk reader
  size_t avg_memory_size;          // average memory usage of this row group

  ChunkInfo() = default;
  std::string ToString() const;
};

struct RowGroupInfo {
  public:
  size_t start_offset;
  size_t end_offset;
  size_t memory_size;

  RowGroupInfo() = default;
  std::string ToString() const;
};

class ColumnGroupReaderInternal;

class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever);

  ~ColumnGroupReaderImpl() = default;

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] size_t total_rows() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(
      const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_size(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  size_t num_of_files_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  size_t total_rows_;

  std::unique_ptr<ColumnGroupReaderInternal> internal_;

};  // ColumnGroupReaderImpl

class ColumnGroupReaderInternal {
  public:
  virtual ~ColumnGroupReaderInternal() = default;

  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::Table>> get_chunk(
      size_t file_index, const std::vector<RowGroupInfo>& row_group_info, const int& rg_index_in_file) = 0;
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::Table>> get_chunks(
      size_t file_index,
      const std::vector<RowGroupInfo>& row_group_info,
      const std::vector<int>& rg_indices_in_file) = 0;

  [[nodiscard]] virtual arrow::Result<std::pair<std::vector<ChunkInfo>, std::vector<std::vector<RowGroupInfo>>>>
  open() = 0;
  static arrow::Result<std::unique_ptr<ColumnGroupReaderInternal>> create(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const milvus_storage::api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever);
};  // ColumnGroupReaderInternal

}  // namespace internal::api
