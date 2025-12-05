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

#include "milvus-storage/format/parquet/parquet_format_reader.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>

#include <arrow/array/util.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace milvus_storage::parquet {

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

static std::vector<RowGroupInfo> create_row_group_infos(const RowGroupMetadataVector& row_group_metadatas) {
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

ParquetFormatReader::ParquetFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                         const std::string& path,
                                         const milvus_storage::api::Properties& properties,
                                         const std::vector<std::string>& needed_columns,
                                         const std::function<std::string(const std::string&)>& key_retriever)
    : path_(path),
      fs_(std::move(fs)),
      schema_(nullptr),
      properties_(std::move(properties)),
      needed_columns_(std::move(needed_columns)),
      key_retriever_(key_retriever),
      file_reader_(nullptr) {}

arrow::Status ParquetFormatReader::open() {
  assert(file_reader_ == nullptr);

  // create file reader
  auto reader_props = ::parquet::default_reader_properties();
  if (key_retriever_) {
    reader_props.file_decryption_properties(::parquet::FileDecryptionProperties::Builder()
                                                .key_retriever(std::make_shared<KeyRetriever>(key_retriever_))
                                                ->plaintext_files_allowed()
                                                ->build());
  }

  auto result = MakeArrowFileReader(*fs_, path_, reader_props);
  if (!result.ok()) {
    return arrow::Status::Invalid("Error making file reader [path= ", std::string(path_),
                                  "] details: ", result.status().ToString());
  }
  file_reader_ = std::move(result.ValueOrDie());

  // create row group infos
  ARROW_ASSIGN_OR_RAISE(auto row_group_metadata,
                        get_row_group_metadata(file_reader_->parquet_reader()->metadata(), path_));

  row_group_infos_ = create_row_group_infos(row_group_metadata);

  // get the schema and create needed column indices
  std::shared_ptr<arrow::Schema> file_schema;
  ARROW_RETURN_NOT_OK(file_reader_->GetSchema(&file_schema));
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

  return arrow::Status::OK();
}

arrow::Result<std::vector<RowGroupInfo>> ParquetFormatReader::get_row_group_infos() { return row_group_infos_; }

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ParquetFormatReader::get_chunk(const int& row_group_index) {
  std::shared_ptr<arrow::Table> table;
  assert(file_reader_);

  if (row_group_index >= row_group_infos_.size()) {
    return arrow::Status::Invalid("Row group index out of range [path=", path_, ", row_group_index=", row_group_index,
                                  ", row_group_infos=", row_group_infos_.size(), "]");
  }

  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroup(row_group_index, needed_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid("Failed to read row group. " + row_group_infos_[row_group_index].ToString());
  }

  return milvus_storage::ConvertTableToRecordBatch(table, false);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetFormatReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  std::unique_ptr<arrow::RecordBatchReader> rb_reader;
  assert(file_reader_);

  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroups(rg_indices_in_file, needed_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid("Failed to read row groups. [path=", path_, "]");
  }
  rb_reader = std::make_unique<arrow::TableBatchReader>(*table);

  std::shared_ptr<arrow::RecordBatch> rb;
  while (true) {
    ARROW_RETURN_NOT_OK(rb_reader->ReadNext(&rb));
    if (!rb) {
      break;
    }
    result.emplace_back(rb);
  }
  ARROW_RETURN_NOT_OK(rb_reader->Close());

  return result;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetFormatReader::take(
    const std::vector<uint64_t>& row_indices) {
  return arrow::Status::NotImplemented("take is not implemented");
}

arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> ParquetFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  std::unique_ptr<arrow::RecordBatchReader> rb_reader;
  if (start_offset >= end_offset) {
    return arrow::Status::Invalid("Invalid range: start_offset=", start_offset, ", end_offset=", end_offset);
  }

  std::vector<int> rg_indices;
  uint64_t current_offset = 0;
  uint64_t first_rg_start_offset = 0;
  bool first_rg_found = false;

  for (size_t i = 0; i < row_group_infos_.size(); ++i) {
    const auto& rg_info = row_group_infos_[i];
    uint64_t rg_start = rg_info.start_offset;
    uint64_t rg_end = rg_info.end_offset;

    if (rg_end > start_offset && rg_start < end_offset) {
      rg_indices.emplace_back(i);
      if (!first_rg_found) {
        first_rg_start_offset = rg_start;
        first_rg_found = true;
      }
    }
  }

  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroups(rg_indices, needed_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid("Failed to read row groups. [path=", path_, "]");
  }

  // Slice the table to the requested range
  int64_t slice_offset = start_offset - first_rg_start_offset;
  int64_t slice_length = end_offset - start_offset;
  auto sliced_table = table->Slice(slice_offset, slice_length);

  rb_reader = std::make_unique<arrow::TableBatchReader>(sliced_table);
  return rb_reader;
}

}  // namespace milvus_storage::parquet