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

#include "milvus-storage/format/parquet/parquet_chunk_reader.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace milvus_storage::parquet {

ParquetChunkReader::ParquetChunkReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                       const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                                       const milvus_storage::api::Properties& properties,
                                       const std::vector<std::string>& needed_columns,
                                       const std::function<std::string(const std::string&)>& key_retriever)
    : fs_(std::move(fs)),
      schema_(nullptr),
      column_group_(column_group),
      cg_files_(column_group->files),
      properties_(std::move(properties)),
      needed_columns_(std::move(needed_columns)),
      key_retriever_(key_retriever),
      file_readers_() {}

static arrow::Result<RowGroupMetadataVector> get_row_group_metadata(
    const std::shared_ptr<::parquet::FileMetaData>& metadata, const std::string& path) {
  assert(metadata);
  if (!metadata) {
    return arrow::Status::Invalid("Failed to get parquet file metadata for file: ", path);
  }

  auto key_value_metadata = metadata->key_value_metadata();
  auto row_group_meta_result = key_value_metadata->Get(ROW_GROUP_META_KEY);
  if (!row_group_meta_result.ok()) {
    return arrow::Status::Invalid("Row group metadata not found, [path=", path,
                                  "], details:", row_group_meta_result.status().ToString());
  }

  return RowGroupMetadataVector::Deserialize(row_group_meta_result.ValueOrDie());
}

static std::vector<RowGroupInfo> get_row_group_infos(const RowGroupMetadataVector& row_group_metadatas) {
  std::vector<RowGroupInfo> row_group_infos(row_group_metadatas.size());
  size_t offset = 0;
  for (size_t i = 0; i < row_group_metadatas.size(); ++i) {
    auto row_group_metadata = row_group_metadatas.Get(i);
    row_group_infos[i] = RowGroupInfo{.start_offset = offset,
                                      .end_offset = offset + row_group_metadata.row_num(),
                                      .memory_size = row_group_metadata.memory_size()};
    offset += row_group_metadata.row_num();
  }

  return row_group_infos;
}

arrow::Result<std::pair<std::vector<ChunkInfo>, std::vector<std::vector<RowGroupInfo>>>> ParquetChunkReader::open() {
  assert(file_readers_.empty());

  // init the parquet reader properties
  reader_props_ = ::parquet::default_reader_properties();
  if (key_retriever_) {
    reader_props_.file_decryption_properties(::parquet::FileDecryptionProperties::Builder()
                                                 .key_retriever(std::make_shared<KeyRetriever>(key_retriever_))
                                                 ->plaintext_files_allowed()
                                                 ->build());
  }

  // Open files and read metadata
  std::vector<ChunkInfo> chunk_infos;
  std::vector<std::vector<RowGroupInfo>> all_row_group_infos;
  size_t file_rows = 0;
  for (size_t i = 0; i < cg_files_.size(); ++i) {
    const auto& cg_file = cg_files_[i];
    const auto& file_path = cg_file.path;
    std::shared_ptr<::parquet::arrow::FileReader> file_reader;
    auto result = MakeArrowFileReader(*fs_, file_path, reader_props_);
    if (!result.ok()) {
      return arrow::Status::Invalid("Error making file reader [path= ", std::string(file_path),
                                    "] details: ", result.status().ToString());
    }
    file_reader = std::move(result.ValueOrDie());
    file_readers_.emplace_back(file_reader);

    // get the RowGroupMetadataVector
    ARROW_ASSIGN_OR_RAISE(auto row_group_metadata,
                          get_row_group_metadata(file_reader->parquet_reader()->metadata(), file_path));

    // get the chunk infos, maybe we need do the pre-cache?
    auto row_group_infos = get_row_group_infos(row_group_metadata);

    // create the chunk infos with start/end indices
    size_t rows_in_file = 0;
    if (cg_file.start_index.has_value() && cg_file.end_index.has_value()) {
      const auto& start_index = cg_file.start_index.value();
      const auto& end_index = cg_file.end_index.value();

      assert(start_index >= 0 && end_index > 0 && start_index < end_index);

      for (size_t j = 0; j < row_group_infos.size(); ++j) {
        size_t rg_start = row_group_infos[j].start_offset;
        size_t rg_end = row_group_infos[j].end_offset;

        // calculate the overlap range
        size_t overlap_start = std::max((size_t)start_index, rg_start);
        size_t overlap_end = std::min((size_t)end_index, rg_end);

        // if the overlap range is valid, create the chunk info
        if (overlap_start < overlap_end) {
          rows_in_file += (overlap_end - overlap_start);
          chunk_infos.emplace_back(ChunkInfo{
              .file_index = i,
              .row_offset_in_row_group = overlap_start - rg_start,
              .row_offset_in_file = overlap_start,
              .number_of_rows = overlap_end - overlap_start,
              .row_group_index_in_file = j,
              .global_row_end = file_rows + rows_in_file,
              .avg_memory_size = row_group_infos[j].memory_size * (overlap_end - overlap_start) / (rg_end - rg_start),
          });
        }
      }
    } else {
      // create the chunk infos with row group indices
      for (size_t j = 0; j < row_group_infos.size(); ++j) {
        rows_in_file += (row_group_infos[j].end_offset - row_group_infos[j].start_offset);
        chunk_infos.emplace_back(ChunkInfo{
            .file_index = i,
            .row_offset_in_row_group = 0,
            .row_offset_in_file = row_group_infos[j].start_offset,
            .number_of_rows = row_group_infos[j].end_offset - row_group_infos[j].start_offset,
            .row_group_index_in_file = j,
            .global_row_end = file_rows + rows_in_file,
            .avg_memory_size = row_group_infos[j].memory_size,
        });
      }
    }
    file_rows += rows_in_file;

    all_row_group_infos.emplace_back(std::move(row_group_infos));
  }

  std::shared_ptr<arrow::Schema> file_schema;
  // Use the first file reader to get schema
  if (file_readers_.empty()) {
    return arrow::Status::Invalid("No parquet files found in column group.");
  }
  ARROW_RETURN_NOT_OK(file_readers_[0]->GetSchema(&file_schema));
  schema_ = file_schema;

  // Convert needed column names to column indices
  std::vector<int> column_indices;
  if (needed_columns_.empty()) {
    for (int i = 0; i < schema_->num_fields(); ++i) {
      column_indices.emplace_back(i);
    }
  } else {
    for (const auto& col_name : needed_columns_) {
      int col_index = schema_->GetFieldIndex(col_name);
      if (col_index >= 0) {
        column_indices.emplace_back(col_index);
      } else {
        return arrow::Status::Invalid("Column " + col_name + " not found in schema");
      }
    }
  }

  needed_column_indices_ = column_indices;
  return std::make_pair(chunk_infos, all_row_group_infos);
}

arrow::Result<std::shared_ptr<arrow::Table>> ParquetChunkReader::get_chunk(
    size_t file_index, const std::vector<RowGroupInfo>& row_group_info, const int& rg_index_in_file) {
  std::shared_ptr<arrow::Table> table;
  assert(!file_readers_.empty());
  assert(file_index < file_readers_.size());
  ARROW_RETURN_NOT_OK(file_readers_[file_index]->ReadRowGroup(rg_index_in_file, needed_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid("Failed to read row group. " + row_group_info[rg_index_in_file].ToString());
  }
  return table;
}

arrow::Result<std::shared_ptr<arrow::Table>> ParquetChunkReader::get_chunks(
    size_t file_index,
    const std::vector<RowGroupInfo>& /*row_group_info*/,
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  assert(file_index < file_readers_.size());
  auto file_reader = file_readers_[file_index];
  assert(file_reader);

  ARROW_RETURN_NOT_OK(file_reader->ReadRowGroups(rg_indices_in_file, needed_column_indices_, &table));
  return table;
}

}  // namespace milvus_storage::parquet
