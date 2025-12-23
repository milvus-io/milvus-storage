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

#include <memory>
#include <optional>
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

static arrow::Result<std::vector<RowGroupInfo>> try_build_row_group_infos(
    const std::shared_ptr<::parquet::FileMetaData>& metadata) {
  std::vector<RowGroupInfo> row_group_infos;
  auto key_value_metadata = metadata->key_value_metadata();
  if (!key_value_metadata) {
    return row_group_infos;
  }

  auto row_group_meta_result = key_value_metadata->Get(ROW_GROUP_META_KEY);
  if (!row_group_meta_result.ok()) {
    return row_group_infos;
  }

  auto row_group_metadatas = RowGroupMetadataVector::Deserialize(row_group_meta_result.ValueOrDie());
  row_group_infos.reserve(row_group_metadatas.size());
  size_t offset = 0;
  for (size_t i = 0; i < row_group_metadatas.size(); ++i) {
    auto row_group_metadata = row_group_metadatas.Get(i);
    row_group_infos.emplace_back(RowGroupInfo{.start_offset = offset,
                                              .end_offset = offset + row_group_metadata.row_num(),
                                              .memory_size = row_group_metadata.memory_size()});
    offset += row_group_metadata.row_num();
  }

  return row_group_infos;
}

arrow::Result<std::vector<RowGroupInfo>> ParquetFormatReader::create_row_group_infos(
    const std::shared_ptr<::parquet::FileMetaData>& metadata) {
  assert(metadata);
  if (!metadata) {
    return arrow::Status::Invalid("Failed to get parquet file metadata for file: ", path_);
  }

  // try use the private kv metas to build row group infos
  ARROW_ASSIGN_OR_RAISE(auto row_group_infos, try_build_row_group_infos(metadata));
  if (!row_group_infos.empty()) {
    return row_group_infos;
  }

  // use the parquet file metadata to build row group infos
  row_group_infos.reserve(metadata->num_row_groups());
  size_t offset = 0;
  for (int i = 0; i < metadata->num_row_groups(); ++i) {
    auto row_group_meta = metadata->RowGroup(i);
    row_group_infos.emplace_back(RowGroupInfo{.start_offset = offset,
                                              .end_offset = offset + static_cast<size_t>(row_group_meta->num_rows()),
                                              .memory_size = static_cast<size_t>(row_group_meta->total_byte_size())});
    offset += row_group_meta->num_rows();
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

static arrow::Result<std::unique_ptr<::parquet::arrow::FileReader>> create_parquet_file_reader(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::string& file_path,
    const std::function<std::string(const std::string&)>& key_retriever,
    std::shared_ptr<::parquet::FileMetaData> metadata = nullptr) {
  std::unique_ptr<::parquet::arrow::FileReader> result;

  ::parquet::arrow::FileReaderBuilder builder;
  ::parquet::ReaderProperties reader_props;
  ::parquet::ArrowReaderProperties arrow_reader_props;

  if (key_retriever) {
    reader_props.file_decryption_properties(::parquet::FileDecryptionProperties::Builder()
                                                .key_retriever(std::make_shared<KeyRetriever>(key_retriever))
                                                ->plaintext_files_allowed()
                                                ->build());
  }
  arrow_reader_props.set_batch_size(INT64_MAX);

  // FIXME(jiaqizho): Although current input no call the close is fine(see ObjectInputFile::Close()),
  // but better to call Close() in function.
  ARROW_ASSIGN_OR_RAISE(auto parquet_file, fs->OpenInputFile(file_path));
  ARROW_RETURN_NOT_OK(builder.Open(std::move(parquet_file), reader_props, metadata));
  ARROW_RETURN_NOT_OK(
      builder.memory_pool(arrow::default_memory_pool())->properties(arrow_reader_props)->Build(&result));
  return std::move(result);
}

arrow::Status ParquetFormatReader::open() {
  assert(file_reader_ == nullptr);

  // create file reader
  ARROW_ASSIGN_OR_RAISE(file_reader_, create_parquet_file_reader(fs_, path_, key_retriever_, nullptr /* metadata */));

  // create row group infos
  assert(file_reader_->parquet_reader() && "arrow logical fault");
  ARROW_ASSIGN_OR_RAISE(row_group_infos_, create_row_group_infos(file_reader_->parquet_reader()->metadata()));

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

arrow::Result<std::shared_ptr<arrow::Table>> ParquetFormatReader::get_chunks_internal(
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  assert(file_reader_);

  ARROW_RETURN_NOT_OK(file_reader_->ReadRowGroups(rg_indices_in_file, needed_column_indices_, &table));

  if (!table) {
    return arrow::Status::Invalid("Failed to read row groups. [path=", path_, "]");
  }
  return table;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ParquetFormatReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  std::shared_ptr<arrow::Table> table;
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  std::unique_ptr<arrow::RecordBatchReader> rb_reader;
  assert(file_reader_);

  ARROW_ASSIGN_OR_RAISE(table, get_chunks_internal(rg_indices_in_file));
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

arrow::Result<std::vector<int>> ParquetFormatReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  std::vector<int> chunk_indices;
  chunk_indices.reserve(row_indices.size());

  for (const auto& row_index : row_indices) {
    auto it = std::upper_bound(row_group_infos_.begin(), row_group_infos_.end(), row_index,
                               [](int64_t val, const RowGroupInfo& info) { return val < info.start_offset; });

    bool found = false;
    if (it != row_group_infos_.begin()) {
      auto prev = std::prev(it);
      if (row_index < prev->end_offset) {
        chunk_indices.emplace_back(std::distance(row_group_infos_.begin(), prev));
        found = true;
      }
    }

    if (!found) {
      return arrow::Status::Invalid("Row index out of range: ", row_index);
    }
  }
  assert(chunk_indices.size() == row_indices.size());

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::Table>> ParquetFormatReader::take(const std::vector<int64_t>& row_indices) {
  ARROW_ASSIGN_OR_RAISE(auto chunk_indices, get_chunk_indices(row_indices));
  assert(chunk_indices.size() == row_indices.size());

  // Deduplicate chunk indices
  auto unique_chunk_indices = chunk_indices;
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  // The input row_indices must be sorted and unique
  ARROW_ASSIGN_OR_RAISE(auto table, get_chunks_internal(unique_chunk_indices));

  // Build a map of chunk_id -> offset in the result table
  std::unordered_map<int, int64_t> chunk_base_offsets;
  int64_t current_accumulated_rows = 0;
  for (int chunk_id : unique_chunk_indices) {
    chunk_base_offsets[chunk_id] = current_accumulated_rows;
    // Accumulate the number of rows for each chunk (end - start)
    const auto& rg_info = row_group_infos_[chunk_id];
    current_accumulated_rows += (rg_info.end_offset - rg_info.start_offset);
  }

  // Calculate take indices for each target row
  std::vector<int64_t> table_take_indices(row_indices.size());
  for (size_t i = 0; i < row_indices.size(); ++i) {
    int chunk_id = chunk_indices[i];
    // Formula: base_offset_in_table + (global_row_index - chunk_start_offset)
    table_take_indices[i] = chunk_base_offsets[chunk_id] + (row_indices[i] - row_group_infos_[chunk_id].start_offset);
  }

  ARROW_ASSIGN_OR_RAISE(table, CopySelectedRows(table, table_take_indices));
  return table;
}

class RangeRecordBatchReader : public arrow::RecordBatchReader {
  public:
  RangeRecordBatchReader(std::unique_ptr<arrow::RecordBatchReader> reader,
                         const uint64_t& first_rg_slice_offset,
                         const uint64_t& total_rows)
      : reader_(std::move(reader)),
        first_rg_slice_offset_(first_rg_slice_offset),
        total_rows_(total_rows),
        current_row_index_(0) {}
  ~RangeRecordBatchReader() override {}

  arrow::Status ReadNext(std::shared_ptr<::arrow::RecordBatch>* out) override {
    assert(current_row_index_ <= total_rows_);
    if (current_row_index_ == total_rows_) {  // no more rows
      *out = nullptr;
      return arrow::Status::OK();
    }

    std::shared_ptr<::arrow::RecordBatch> rb;
    ARROW_RETURN_NOT_OK(reader_->ReadNext(&rb));

    // first row group
    if (current_row_index_ == 0) {
      if (rb->num_rows() < first_rg_slice_offset_) {
        return arrow::Status::Invalid("Logical error, first_rg_slice_offset_=", first_rg_slice_offset_,
                                      ", the first row group has ", rb->num_rows(), " rows");
      }

      // current range exist in the first row group
      if (rb->num_rows() - first_rg_slice_offset_ >= total_rows_) {
        *out = rb->Slice(first_rg_slice_offset_, total_rows_);
        current_row_index_ += total_rows_;
      } else {  // still need read next row group
        *out = rb->Slice(first_rg_slice_offset_, rb->num_rows() - first_rg_slice_offset_);
        current_row_index_ += rb->num_rows() - first_rg_slice_offset_;
      }

    } else {  // not the first row group
      uint64_t remain_rows = total_rows_ - current_row_index_;
      if (remain_rows >= rb->num_rows()) {  // still need read next row group
        *out = rb;
        current_row_index_ += rb->num_rows();
      } else {  // no more row groups
        *out = rb->Slice(0, remain_rows);
        current_row_index_ += remain_rows;
      }
    }

    return arrow::Status::OK();
  }

  std::shared_ptr<::arrow::Schema> schema() const override { return reader_->schema(); }

  private:
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  uint64_t first_rg_slice_offset_;
  uint64_t total_rows_;
  uint64_t current_row_index_;
};

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ParquetFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  std::unique_ptr<arrow::RecordBatchReader> rb_reader;
  if (row_group_infos_.empty()) {
    return arrow::Status::Invalid("Empty row group infos");
  }

  if (start_offset >= end_offset || start_offset < row_group_infos_.front().start_offset ||
      end_offset > row_group_infos_.back().end_offset) {
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

  assert(end_offset <= row_group_infos_[rg_indices.back()].end_offset);
  uint64_t first_rg_slice_offset = start_offset - first_rg_start_offset;
  uint64_t last_rg_slice_offset = end_offset - row_group_infos_[rg_indices.back()].end_offset;
  uint64_t total_rows = end_offset - start_offset;

  ARROW_RETURN_NOT_OK(file_reader_->GetRecordBatchReader(rg_indices, needed_column_indices_, &rb_reader));

  // no need slice
  if (first_rg_slice_offset == 0 && last_rg_slice_offset == 0) {
    return rb_reader;
  }

  return std::make_shared<RangeRecordBatchReader>(std::move(rb_reader), first_rg_slice_offset, total_rows);
}

arrow::Result<std::shared_ptr<FormatReader>> ParquetFormatReader::clone_reader() {
  assert(file_reader_);

  ARROW_ASSIGN_OR_RAISE(auto parquet_reader, create_parquet_file_reader(fs_, path_, key_retriever_,
                                                                        file_reader_->parquet_reader()->metadata()));
  return std::shared_ptr<ParquetFormatReader>(new ParquetFormatReader(*this, std::move(parquet_reader)));
}

ParquetFormatReader::ParquetFormatReader(const ParquetFormatReader& other,
                                         std::unique_ptr<::parquet::arrow::FileReader> cloned_file_reader)
    : path_(other.path_),
      fs_(other.fs_),
      schema_(other.schema_),
      properties_(other.properties_),
      needed_columns_(other.needed_columns_),
      key_retriever_(other.key_retriever_),
      needed_column_indices_(other.needed_column_indices_),
      row_group_infos_(other.row_group_infos_),
      file_reader_(std::move(cloned_file_reader)) {}

}  // namespace milvus_storage::parquet