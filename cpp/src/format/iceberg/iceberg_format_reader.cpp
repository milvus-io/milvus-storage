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

#include "milvus-storage/format/iceberg/iceberg_format_reader.h"

#include <algorithm>
#include <string>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <fmt/format.h>
#include <folly/json/json.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::iceberg {

IcebergFormatReader::IcebergFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                         const std::string& resolved_path,
                                         const std::string& data_file_uri,
                                         const std::vector<uint8_t>& delete_metadata,
                                         const api::Properties& properties,
                                         const std::vector<std::string>& needed_columns,
                                         const std::function<std::string(const std::string&)>& key_retriever)
    : inner_reader_(
          std::make_shared<parquet::ParquetFormatReader>(fs, resolved_path, properties, needed_columns, key_retriever)),
      data_file_uri_(data_file_uri),
      delete_metadata_(delete_metadata),
      properties_(properties),
      deleted_positions_(std::make_shared<std::unordered_set<int64_t>>()) {}

IcebergFormatReader::IcebergFormatReader(std::shared_ptr<parquet::ParquetFormatReader> inner,
                                         const std::string& data_file_uri,
                                         const api::Properties& properties,
                                         std::shared_ptr<std::unordered_set<int64_t>> deleted_positions)
    : inner_reader_(std::move(inner)),
      data_file_uri_(data_file_uri),
      properties_(properties),
      deleted_positions_(std::move(deleted_positions)) {}

arrow::Status IcebergFormatReader::open() {
  ARROW_RETURN_NOT_OK(inner_reader_->open());
  return load_positional_deletes();
}

arrow::Status IcebergFormatReader::load_positional_deletes() {
  if (delete_metadata_.empty()) {
    return arrow::Status::OK();
  }

  std::string json_str(delete_metadata_.begin(), delete_metadata_.end());
  folly::dynamic parsed;
  try {
    parsed = folly::parseJson(json_str);
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to parse delete metadata JSON: {}", e.what()));
  }

  if (!parsed.isArray()) {
    return arrow::Status::Invalid("Delete metadata JSON must be an array");
  }

  for (const auto& entry : parsed) {
    auto file_type = entry.getDefault("file_type", "").asString();
    auto path = entry.getDefault("path", "").asString();

    if (file_type == "equality") {
      return arrow::Status::Invalid(
          fmt::format("Equality deletes not supported at read time. Delete file: {}. "
                      "Equality deletes must be converted to positional deletes before explore.",
                      path));
    }

    if (file_type == "position") {
      ARROW_RETURN_NOT_OK(read_positional_delete_file(path));
    }
    // Skip unknown types (forward compatibility for deletion vectors, etc.)
  }

  return arrow::Status::OK();
}

arrow::Status IcebergFormatReader::read_positional_delete_file(const std::string& delete_file_path) {
  // Get filesystem for the delete file
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties_, delete_file_path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(delete_file_path));
  std::string resolved_path = uri.scheme.empty() ? delete_file_path : uri.key;

  // Open the delete file as Parquet
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(resolved_path));

  ::parquet::arrow::FileReaderBuilder builder;
  std::unique_ptr<::parquet::arrow::FileReader> file_reader;
  ARROW_RETURN_NOT_OK(builder.Open(std::move(input_file)));
  ARROW_RETURN_NOT_OK(builder.memory_pool(arrow::default_memory_pool())->Build(&file_reader));

  // Read all row groups into a table with columns: file_path, pos
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(file_reader->GetSchema(&schema));

  int file_path_idx = schema->GetFieldIndex("file_path");
  int pos_idx = schema->GetFieldIndex("pos");

  if (file_path_idx < 0 || pos_idx < 0) {
    return arrow::Status::Invalid(
        fmt::format("Positional delete file missing required columns (file_path, pos): {}", delete_file_path));
  }

  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(file_reader->ReadTable({file_path_idx, pos_idx}, &table));

  // Filter rows where file_path matches our data file and collect pos values
  auto file_path_col = table->column(0);  // file_path (first in our projection)
  auto pos_col = table->column(1);        // pos (second in our projection)

  for (int chunk_idx = 0; chunk_idx < file_path_col->num_chunks(); ++chunk_idx) {
    auto file_path_array = std::static_pointer_cast<arrow::StringArray>(file_path_col->chunk(chunk_idx));
    auto pos_array = std::static_pointer_cast<arrow::Int64Array>(pos_col->chunk(chunk_idx));

    for (int64_t row = 0; row < file_path_array->length(); ++row) {
      if (!file_path_array->IsNull(row) && file_path_array->GetView(row) == data_file_uri_) {
        if (!pos_array->IsNull(row)) {
          deleted_positions_->insert(pos_array->Value(row));
        }
      }
    }
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> IcebergFormatReader::filter_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch, size_t chunk_start) {
  if (deleted_positions_->empty()) {
    return batch;
  }

  // Build keep-indices: positions NOT in the deletion set
  std::vector<int64_t> keep_indices;
  keep_indices.reserve(batch->num_rows());
  for (int64_t i = 0; i < batch->num_rows(); ++i) {
    int64_t global_pos = static_cast<int64_t>(chunk_start) + i;
    if (deleted_positions_->find(global_pos) == deleted_positions_->end()) {
      keep_indices.push_back(i);
    }
  }

  if (static_cast<int64_t>(keep_indices.size()) == batch->num_rows()) {
    return batch;  // No rows deleted in this chunk
  }

  // Use arrow::compute::Take to select kept rows
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(keep_indices.size()));
  ARROW_RETURN_NOT_OK(builder.AppendValues(keep_indices));
  ARROW_ASSIGN_OR_RAISE(auto indices_array, builder.Finish());

  ARROW_ASSIGN_OR_RAISE(auto result, arrow::compute::Take(batch, indices_array));
  return result.record_batch();
}

arrow::Result<std::vector<RowGroupInfo>> IcebergFormatReader::get_row_group_infos() {
  return inner_reader_->get_row_group_infos();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> IcebergFormatReader::get_chunk(const int& row_group_index) {
  ARROW_ASSIGN_OR_RAISE(auto batch, inner_reader_->get_chunk(row_group_index));

  if (deleted_positions_->empty()) {
    return batch;
  }

  // Determine the global physical offset for this row group
  ARROW_ASSIGN_OR_RAISE(auto rg_infos, inner_reader_->get_row_group_infos());
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= rg_infos.size()) {
    return arrow::Status::Invalid(fmt::format("Row group index out of range: {}", row_group_index));
  }

  return filter_batch(batch, rg_infos[row_group_index].start_offset);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> IcebergFormatReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  ARROW_ASSIGN_OR_RAISE(auto batches, inner_reader_->get_chunks(rg_indices_in_file));

  if (deleted_positions_->empty()) {
    return batches;
  }

  ARROW_ASSIGN_OR_RAISE(auto rg_infos, inner_reader_->get_row_group_infos());

  std::vector<std::shared_ptr<arrow::RecordBatch>> filtered;
  filtered.reserve(batches.size());
  for (size_t i = 0; i < batches.size(); ++i) {
    int rg_idx = rg_indices_in_file[i];
    ARROW_ASSIGN_OR_RAISE(auto fb, filter_batch(batches[i], rg_infos[rg_idx].start_offset));
    filtered.push_back(std::move(fb));
  }

  return filtered;
}

arrow::Result<std::shared_ptr<arrow::Table>> IcebergFormatReader::take(const std::vector<int64_t>& row_indices) {
  if (deleted_positions_->empty()) {
    return inner_reader_->take(row_indices);
  }

  // Build sorted deletion vector for binary search
  std::vector<int64_t> sorted_dels(deleted_positions_->begin(), deleted_positions_->end());
  std::sort(sorted_dels.begin(), sorted_dels.end());

  // Map logical indices (post-delete doc IDs) to physical indices (pre-delete).
  // Index building sees data without deleted rows, so logical index N refers to
  // the (N+1)-th non-deleted row. We find the physical position p such that
  // p - (number of deletions <= p) == logical_index.
  std::vector<int64_t> physical;
  physical.reserve(row_indices.size());
  for (auto logical_idx : row_indices) {
    int64_t p = logical_idx;
    while (true) {
      auto num_dels =
          static_cast<int64_t>(std::upper_bound(sorted_dels.begin(), sorted_dels.end(), p) - sorted_dels.begin());
      int64_t new_p = logical_idx + num_dels;
      if (new_p == p) {
        break;
      }
      p = new_p;
    }
    physical.push_back(p);
  }

  return inner_reader_->take(physical);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> IcebergFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  // Delegate directly. Filtering at the read_with_range level would
  // break ColumnGroupReaderImpl's offset model. Callers that need
  // delete-aware reads should use get_chunk() or take().
  return inner_reader_->read_with_range(start_offset, end_offset);
}

arrow::Result<std::shared_ptr<FormatReader>> IcebergFormatReader::clone_reader() {
  ARROW_ASSIGN_OR_RAISE(auto cloned_inner, inner_reader_->clone_reader());
  auto cloned_parquet = std::dynamic_pointer_cast<parquet::ParquetFormatReader>(cloned_inner);
  if (!cloned_parquet) {
    return arrow::Status::Invalid("Failed to clone inner ParquetFormatReader");
  }

  return std::shared_ptr<FormatReader>(
      new IcebergFormatReader(std::move(cloned_parquet), data_file_uri_, properties_, deleted_positions_));
}

}  // namespace milvus_storage::iceberg
