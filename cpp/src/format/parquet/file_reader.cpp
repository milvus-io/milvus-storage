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

#include <memory>
#include <numeric>
#include <string>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/logging.h>

#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>

#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/packed/chunk_manager.h"

namespace milvus_storage {

FileRowGroupReader::FileRowGroupReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       const std::string& path,
                                       const int64_t buffer_size,
                                       parquet::ReaderProperties reader_props) {
  auto status = init(fs, path, buffer_size, nullptr, reader_props);
  if (!status.ok()) {
    ARROW_LOG(ERROR) << "Error initializing file reader: " << status.ToString();
    throw std::runtime_error(status.ToString());
  }
}

FileRowGroupReader::FileRowGroupReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                       const std::string& path,
                                       const std::shared_ptr<arrow::Schema> schema,
                                       const int64_t buffer_size,
                                       parquet::ReaderProperties reader_props) {
  auto status = init(fs, path, buffer_size, schema, reader_props);
  if (!status.ok()) {
    ARROW_LOG(ERROR) << "Error initializing file reader: " << status.ToString();
    throw std::runtime_error(status.ToString());
  }
}

Status FileRowGroupReader::init(std::shared_ptr<arrow::fs::FileSystem> fs,
                                const std::string& path,
                                const int64_t buffer_size,
                                const std::shared_ptr<arrow::Schema> schema,
                                parquet::ReaderProperties reader_props) {
  fs_ = std::move(fs);
  path_ = path;
  buffer_size_limit_ = buffer_size <= 0 ? INT64_MAX : buffer_size;

  // Open the file
  auto result = MakeArrowFileReader(*fs_, path_, reader_props);
  if (!result.ok()) {
    return Status::ReaderError("Error making file reader:" + result.status().ToString());
  }
  file_reader_ = std::move(result.value());

  auto metadata = file_reader_->parquet_reader()->metadata();
  ASSIGN_OR_RETURN_NOT_OK(file_metadata_, PackedFileMetadata::Make(metadata));

  // If schema is not provided, use the schema from the file
  if (schema == nullptr) {
    std::shared_ptr<arrow::Schema> file_schema;
    auto status = file_reader_->GetSchema(&file_schema);
    if (!status.ok()) {
      return Status::ReaderError("Failed to get schema from file: " + status.ToString());
    }
    schema_ = file_schema;
    field_id_list_ = FieldIDList::Make(schema_).value();
    for (size_t i = 0; i < field_id_list_.size(); ++i) {
      needed_columns_.emplace_back(i);
    }
  } else {
    // schema matching
    std::map<FieldID, ColumnOffset> field_id_mapping = file_metadata_->GetFieldIDMapping();
    Result<FieldIDList> status = FieldIDList::Make(schema);
    if (!status.ok()) {
      return Status::MetadataParseError("Error getting field id list from schema: " + schema->ToString());
    }
    field_id_list_ = status.value();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (size_t i = 0; i < field_id_list_.size(); ++i) {
      FieldID field_id = field_id_list_.Get(i);
      if (field_id_mapping.find(field_id) != field_id_mapping.end()) {
        needed_columns_.emplace_back(field_id_mapping[field_id].col_index);
        fields.emplace_back(schema->field(i));
      } else {
        // mark nullable if the field can not be found in the file, in case the reader schema is not marked
        fields.emplace_back(schema->field(i)->WithNullable(true));
      }
    }
    schema_ = std::make_shared<arrow::Schema>(fields);
  }

  return Status::OK();
}

std::shared_ptr<PackedFileMetadata> FileRowGroupReader::file_metadata() { return file_metadata_; }

std::shared_ptr<arrow::Schema> FileRowGroupReader::schema() const { return schema_; }

Status FileRowGroupReader::SetRowGroupOffsetAndCount(int row_group_offset, int row_group_num) {
  if (row_group_offset < 0 || row_group_num <= 0) {
    return Status::InvalidArgument("please provide row group offset and row group num");
  }
  size_t total_row_groups = file_metadata_->GetRowGroupMetadataVector().size();
  if (row_group_offset >= total_row_groups || row_group_offset + row_group_num > total_row_groups) {
    std::string error_msg = "Row group range exceeds total number of row groups: " + std::to_string(total_row_groups);
    return Status::InvalidArgument(error_msg);
  }
  rg_start_ = row_group_offset;
  current_rg_ = row_group_offset;
  rg_end_ = row_group_offset + row_group_num - 1;

  // Clear buffer when resetting row group range to avoid mixing data from different ranges
  buffer_table_ = nullptr;
  buffer_size_ = 0;

  return Status::OK();
}

// Helper function to match schema and fill null columns
void MatchSchemaAndFillNullColumns(const std::shared_ptr<arrow::Table>& table,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const FieldIDList& field_id_list,
                                   const std::map<FieldID, ColumnOffset>& field_id_mapping,
                                   std::shared_ptr<arrow::Table>* out) {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

  for (size_t i = 0; i < field_id_list.size(); ++i) {
    FieldID field_id = field_id_list.Get(i);
    if (field_id_mapping.find(field_id) != field_id_mapping.end()) {
      int col = field_id_mapping.at(field_id).col_index;
      columns.emplace_back(table->column(col));
    } else {
      auto null_array = arrow::MakeArrayOfNull(schema->field(i)->type(), table->num_rows()).ValueOrDie();
      columns.emplace_back(std::make_shared<arrow::ChunkedArray>(null_array));
    }
  }

  *out = arrow::Table::Make(schema, columns);
}

arrow::Status FileRowGroupReader::SliceRowGroupFromTable(std::shared_ptr<arrow::Table>* out) {
  assert(buffer_table_ != nullptr);
  auto row_group_num = file_metadata_->GetRowGroupMetadataVector().Get(current_rg_).row_num();
  assert(buffer_table_->num_rows() >= row_group_num);
  *out = buffer_table_->Slice(0, row_group_num);
  if (buffer_table_->num_rows() == row_group_num) {
    buffer_table_ = nullptr;
    buffer_size_ = 0;
  } else {
    buffer_table_ = buffer_table_->Slice(row_group_num);
    auto new_size = GetTableMemorySize(buffer_table_);
    buffer_size_ = std::max<int64_t>(0, buffer_size_ - new_size);
  }
  current_rg_++;
  return arrow::Status::OK();
}

arrow::Status FileRowGroupReader::ReadNextRowGroup(std::shared_ptr<arrow::Table>* out) {
  if (current_rg_ > rg_end_ || rg_start_ == -1) {
    ARROW_LOG(WARNING) << "Please set row group offset and count before reading next.";
    current_rg_ = -1;
    rg_start_ = -1;
    rg_end_ = -1;
    *out = nullptr;
    return arrow::Status::OK();
  }

  // If buffer table has enough rows, slice with the number of rows in the current row group and return it
  auto row_group_num = file_metadata_->GetRowGroupMetadataVector().Get(current_rg_).row_num();
  if (buffer_table_ != nullptr && buffer_table_->num_rows() >= row_group_num) {
    return SliceRowGroupFromTable(out);
  }

  // Calculate how many row groups we can read with remaining memory
  std::vector<int> rgs_to_read;
  int64_t remaining_memory = buffer_size_limit_ - buffer_size_;
  int rg = rg_start_;

  while (rg <= rg_end_ && remaining_memory >= file_metadata_->GetRowGroupMetadataVector().Get(rg).memory_size()) {
    rgs_to_read.emplace_back(rg);
    remaining_memory -= file_metadata_->GetRowGroupMetadataVector().Get(rg).memory_size();
    rg++;
  }

  // If no row groups can fit in memory, still try to read at least one row group
  if (rgs_to_read.empty() && rg <= rg_end_) {
    // Force read at least one row group
    rgs_to_read.emplace_back(rg);
    rg++;
  }

  if (rgs_to_read.empty()) {
    // No more row groups to read
    if (buffer_table_ != nullptr) {
      std::string error_msg = "No more row groups to read, but buffer table is not empty";
      ARROW_LOG(ERROR) << error_msg;
      return arrow::Status::IOError(error_msg);
    }
    rg_start_ = -1;
    rg_end_ = -1;
    current_rg_ = -1;
    *out = nullptr;
    return arrow::Status::OK();
  }

  // Read new row groups
  std::shared_ptr<arrow::Table> new_table = nullptr;
  auto status = file_reader_->ReadRowGroups(rgs_to_read, needed_columns_, &new_table);
  if (!status.ok()) {
    *out = nullptr;
    return status;
  }

  // Match schema and fill null columns
  std::shared_ptr<arrow::Table> matched_table;
  MatchSchemaAndFillNullColumns(new_table, schema_, field_id_list_, file_metadata_->GetFieldIDMapping(),
                                &matched_table);

  // Merge with existing buffer table if needed
  if (buffer_table_ != nullptr) {
    std::vector<std::shared_ptr<arrow::Table>> tables = {buffer_table_, matched_table};
    auto merged_table = arrow::ConcatenateTables(tables);
    if (!merged_table.ok()) {
      return merged_table.status();
    }
    buffer_table_ = merged_table.ValueOrDie();
  } else {
    buffer_table_ = matched_table;
  }

  buffer_size_ = GetTableMemorySize(buffer_table_);
  rg_start_ = rg;

  return SliceRowGroupFromTable(out);
}

arrow::Status FileRowGroupReader::Close() {
  file_reader_ = nullptr;
  return arrow::Status::OK();
}

ParquetFileReader::ParquetFileReader(std::unique_ptr<parquet::arrow::FileReader> reader) : reader_(std::move(reader)) {}

Result<std::shared_ptr<arrow::RecordBatch>> GetRecordAtOffset(arrow::RecordBatchReader* reader, int64_t offset) {
  int64_t skipped = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  do {
    RETURN_ARROW_NOT_OK(reader->ReadNext(&batch));
    skipped += batch->num_rows();
  } while (skipped < offset);

  auto offset_batch = offset - skipped + batch->num_rows();
  // zero-copy slice
  return batch->Slice(offset_batch, 1);
}

// TODO: support projection
Result<std::shared_ptr<arrow::Table>> ParquetFileReader::ReadByOffsets(std::vector<int64_t>& offsets) {
  std::sort(offsets.begin(), offsets.end());

  auto num_row_groups = reader_->parquet_reader()->metadata()->num_row_groups();
  int rg_idx = 0;
  int64_t total_skipped = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::unique_ptr<arrow::RecordBatchReader> rg_reader;

  for (auto& offset : offsets) {
    // skip row groups
    // TODO: to make read more efficient, we should find offsets belonged to a row group and read together.
    while (rg_idx < num_row_groups) {
      auto row_group_meta = reader_->parquet_reader()->metadata()->RowGroup(rg_idx);
      auto row_group_num_rows = row_group_meta->num_rows();
      if (row_group_num_rows + total_skipped > offset) {
        break;
      }
      rg_idx++;
      total_skipped += row_group_num_rows;
      rg_reader = nullptr;
    }

    if (rg_idx >= num_row_groups) {
      break;
    }

    if (rg_reader == nullptr) {
      RETURN_ARROW_NOT_OK(reader_->GetRecordBatchReader({rg_idx}, &rg_reader));
    }

    auto row_group_offset = offset - total_skipped;
    ASSIGN_OR_RETURN_NOT_OK(auto batch, GetRecordAtOffset(rg_reader.get(), row_group_offset))
    batches.emplace_back(batch);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(batches));
  return res;
}
}  // namespace milvus_storage
