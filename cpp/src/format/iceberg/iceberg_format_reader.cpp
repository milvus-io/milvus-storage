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
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <fmt/format.h>
#include <folly/json/json.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"

namespace milvus_storage::iceberg {
namespace {

std::vector<uint8_t> GetDeleteMetadataBytes(const api::ColumnGroupFile& file) {
  auto metadata_it = file.properties.find(api::kPropertyMetadata);
  if (metadata_it == file.properties.end()) {
    return {};
  }
  return std::vector<uint8_t>(metadata_it->second.begin(), metadata_it->second.end());
}

std::shared_ptr<const std::vector<int64_t>> MakeSortedDeletions(
    const std::shared_ptr<const std::unordered_set<int64_t>>& deleted_positions) {
  auto sorted_deletions = std::make_shared<std::vector<int64_t>>();
  if (deleted_positions) {
    sorted_deletions->assign(deleted_positions->begin(), deleted_positions->end());
    std::sort(sorted_deletions->begin(), sorted_deletions->end());
  }
  return sorted_deletions;
}

std::shared_ptr<const std::unordered_set<int64_t>> EmptyDeletedPositions() {
  return std::make_shared<std::unordered_set<int64_t>>();
}

std::shared_ptr<const std::vector<int64_t>> EmptySortedDeletions() { return std::make_shared<std::vector<int64_t>>(); }

uint64_t EstimateDeleteChargeBytes(const std::vector<uint8_t>& delete_metadata,
                                   const std::unordered_set<int64_t>& deleted_positions,
                                   const std::vector<int64_t>& sorted_deletions,
                                   const std::vector<RowGroupInfo>& logical_row_group_infos) {
  return delete_metadata.size() + deleted_positions.size() * (sizeof(int64_t) + sizeof(void*)) +
         sorted_deletions.size() * sizeof(int64_t) + logical_row_group_infos.size() * sizeof(RowGroupInfo);
}

arrow::Status ReadPositionalDeleteFile(const std::string& delete_file_path,
                                       const std::string& data_file_uri,
                                       const api::Properties& properties,
                                       std::unordered_set<int64_t>* deleted_positions) {
  if (!deleted_positions) {
    return arrow::Status::Invalid("Deleted positions output cannot be null");
  }

  // Get filesystem for the delete file
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, delete_file_path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(delete_file_path));

  // Open the delete file as Parquet
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(uri.key));

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

  // Normalize data_file_uri to scheme://bucket/key for matching.
  // data_file_uri may be in Milvus format (scheme://address/bucket/key),
  // while the delete file's file_path column may use ABFSS opendal format
  // (abfss://container@endpoint/key). MilvusURIToIcebergURI normalizes both
  // to the same scheme://bucket/key form.
  auto normalized_data_uri = MilvusURIToIcebergURI(data_file_uri);

  // Filter rows where file_path matches our data file and collect pos values
  auto file_path_col = table->column(0);  // file_path (first in our projection)
  auto pos_col = table->column(1);        // pos (second in our projection)

  for (int chunk_idx = 0; chunk_idx < file_path_col->num_chunks(); ++chunk_idx) {
    auto file_path_array = std::static_pointer_cast<arrow::StringArray>(file_path_col->chunk(chunk_idx));
    auto pos_array = std::static_pointer_cast<arrow::Int64Array>(pos_col->chunk(chunk_idx));

    for (int64_t row = 0; row < file_path_array->length(); ++row) {
      if (!file_path_array->IsNull(row)) {
        std::string row_file_path(file_path_array->GetView(row));
        // Match by: 1) exact match with original URI,
        //           2) direct match after Milvus address stripping (S3/GCS),
        //           3) match after ABFSS @endpoint stripping (Azure).
        if (row_file_path == data_file_uri || row_file_path == normalized_data_uri ||
            StripAbfssEndpoint(row_file_path) == normalized_data_uri) {
          if (!pos_array->IsNull(row)) {
            deleted_positions->insert(pos_array->Value(row));
          }
        }
      }
    }
  }

  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<const std::unordered_set<int64_t>>> LoadPositionalDeletes(
    const std::vector<uint8_t>& delete_metadata, const std::string& data_file_uri, const api::Properties& properties) {
  auto deleted_positions = std::make_shared<std::unordered_set<int64_t>>();
  if (delete_metadata.empty()) {
    return std::static_pointer_cast<const std::unordered_set<int64_t>>(deleted_positions);
  }

  std::string json_str(delete_metadata.begin(), delete_metadata.end());
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
      ARROW_RETURN_NOT_OK(ReadPositionalDeleteFile(path, data_file_uri, properties, deleted_positions.get()));
    }
    // Skip unknown types (forward compatibility for deletion vectors, etc.)
  }

  return std::static_pointer_cast<const std::unordered_set<int64_t>>(deleted_positions);
}

}  // namespace

std::string IcebergFormatReader::MetaTrait::cache_key(const api::ColumnGroupFile& file) {
  const auto file_size = file.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = file.Get<uint64_t>(api::kPropertyFooterSize);

  std::string key = fmt::format("iceberg:path={};file_size={};footer_size={}", file.path, file_size, footer_size);
  auto metadata_it = file.properties.find(api::kPropertyMetadata);
  if (metadata_it != file.properties.end()) {
    key += fmt::format(";delete_metadata_size={};delete_metadata={}", metadata_it->second.size(), metadata_it->second);
  }
  return key;
}

arrow::Result<IcebergFormatReader::MetaTrait::MetadataPtr> IcebergFormatReader::MetaTrait::load_metadata(
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const milvus_storage::KeyRetriever& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto data_metadata,
                        parquet::ParquetFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
  if (!data_metadata) {
    return arrow::Status::Invalid(fmt::format("Failed to load Iceberg data metadata. [path={}]", file.path));
  }

  auto delete_metadata = GetDeleteMetadataBytes(file);
  ARROW_ASSIGN_OR_RAISE(auto deleted_positions, LoadPositionalDeletes(delete_metadata, file.path, properties));
  auto sorted_deletions = MakeSortedDeletions(deleted_positions);
  ARROW_ASSIGN_OR_RAISE(auto logical_row_group_infos,
                        build_logical_row_group_infos(data_metadata->row_group_infos, *sorted_deletions));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = data_metadata->path;
  metadata->file_schema = data_metadata->file_schema;
  metadata->row_group_infos = std::move(logical_row_group_infos);
  metadata->cache_size =
      data_metadata->cache_size +
      EstimateDeleteChargeBytes(delete_metadata, *deleted_positions, *sorted_deletions, metadata->row_group_infos);
  metadata->payload.data_metadata = std::move(data_metadata);
  metadata->payload.data_file_uri = file.path;
  metadata->payload.delete_metadata = std::move(delete_metadata);
  metadata->payload.properties = properties;
  metadata->payload.deleted_positions = std::move(deleted_positions);
  metadata->payload.sorted_deletions = std::move(sorted_deletions);

  MetadataPtr metadata_ptr = std::move(metadata);
  return metadata_ptr;
}

arrow::Result<std::shared_ptr<IcebergFormatReader>> IcebergFormatReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  if (!metadata) {
    return arrow::Status::Invalid("Cannot open Iceberg reader from null metadata");
  }
  if (!metadata->payload.data_metadata) {
    return arrow::Status::Invalid(
        fmt::format("Cannot open Iceberg reader from metadata without Parquet metadata. [path={}]", metadata->path));
  }

  ARROW_ASSIGN_OR_RAISE(auto inner_reader,
                        parquet::ParquetFormatReader::MetaTrait::create_from_metadata(
                            metadata->payload.data_metadata, file, read_schema, needed_columns, predicate));
  auto deleted_positions =
      metadata->payload.deleted_positions ? metadata->payload.deleted_positions : EmptyDeletedPositions();
  auto sorted_deletions =
      metadata->payload.sorted_deletions ? metadata->payload.sorted_deletions : MakeSortedDeletions(deleted_positions);
  ARROW_ASSIGN_OR_RAISE(auto projected_schema, build_projected_schema(metadata->file_schema, needed_columns));

  return std::shared_ptr<IcebergFormatReader>(new IcebergFormatReader(
      std::move(inner_reader), metadata->payload.data_file_uri, metadata->payload.properties,
      metadata->payload.delete_metadata, std::move(deleted_positions), std::move(sorted_deletions),
      metadata->row_group_infos, needed_columns, std::move(projected_schema)));
}

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
      needed_columns_(needed_columns),
      projected_schema_(nullptr),
      deleted_positions_(EmptyDeletedPositions()),
      sorted_deletions_(EmptySortedDeletions()) {}

IcebergFormatReader::IcebergFormatReader(std::shared_ptr<parquet::ParquetFormatReader> inner,
                                         const std::string& data_file_uri,
                                         const api::Properties& properties,
                                         std::vector<uint8_t> delete_metadata,
                                         std::shared_ptr<const std::unordered_set<int64_t>> deleted_positions,
                                         std::shared_ptr<const std::vector<int64_t>> sorted_deletions,
                                         std::vector<RowGroupInfo> logical_row_group_infos,
                                         std::vector<std::string> needed_columns,
                                         std::shared_ptr<arrow::Schema> projected_schema)
    : inner_reader_(std::move(inner)),
      data_file_uri_(data_file_uri),
      delete_metadata_(std::move(delete_metadata)),
      properties_(properties),
      needed_columns_(std::move(needed_columns)),
      projected_schema_(std::move(projected_schema)),
      deleted_positions_(deleted_positions ? std::move(deleted_positions) : EmptyDeletedPositions()),
      sorted_deletions_(sorted_deletions ? std::move(sorted_deletions) : MakeSortedDeletions(deleted_positions_)),
      logical_row_group_infos_(std::move(logical_row_group_infos)) {}

arrow::Status IcebergFormatReader::open() {
  ARROW_RETURN_NOT_OK(inner_reader_->open());
  ARROW_ASSIGN_OR_RAISE(projected_schema_, build_projected_schema(inner_reader_->get_schema(), needed_columns_));
  ARROW_ASSIGN_OR_RAISE(deleted_positions_, load_positional_deletes());
  sorted_deletions_ = MakeSortedDeletions(deleted_positions_);

  ARROW_ASSIGN_OR_RAISE(auto physical_row_group_infos, inner_reader_->get_row_group_infos());
  ARROW_ASSIGN_OR_RAISE(logical_row_group_infos_,
                        build_logical_row_group_infos(physical_row_group_infos, *sorted_deletions_));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<const std::unordered_set<int64_t>>> IcebergFormatReader::load_positional_deletes() const {
  return LoadPositionalDeletes(delete_metadata_, data_file_uri_, properties_);
}

arrow::Result<std::shared_ptr<arrow::Schema>> IcebergFormatReader::build_projected_schema(
    const std::shared_ptr<arrow::Schema>& file_schema, const std::vector<std::string>& needed_columns) {
  if (!file_schema) {
    return arrow::Status::Invalid("Cannot build Iceberg projected schema from null file schema");
  }
  if (needed_columns.empty()) {
    return file_schema;
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(needed_columns.size());
  for (const auto& column : needed_columns) {
    auto index = file_schema->GetFieldIndex(column);
    if (index < 0) {
      return arrow::Status::Invalid(fmt::format("Column '{}' not found in Iceberg schema", column));
    }
    fields.emplace_back(file_schema->field(index));
  }
  return arrow::schema(std::move(fields));
}

std::shared_ptr<arrow::Schema> IcebergFormatReader::output_schema() const {
  return projected_schema_ ? projected_schema_ : inner_reader_->get_schema();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> IcebergFormatReader::filter_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch, size_t chunk_start) {
  if (!deleted_positions_ || deleted_positions_->empty()) {
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

arrow::Result<std::vector<RowGroupInfo>> IcebergFormatReader::build_logical_row_group_infos(
    const std::vector<RowGroupInfo>& physical_row_group_infos, const std::vector<int64_t>& sorted_deletions) {
  if (sorted_deletions.empty()) {
    return physical_row_group_infos;
  }

  std::vector<RowGroupInfo> logical_row_group_infos;
  logical_row_group_infos.reserve(physical_row_group_infos.size());

  uint64_t logical_offset = 0;
  for (const auto& prg : physical_row_group_infos) {
    if (prg.end_offset < prg.start_offset) {
      return arrow::Status::Invalid(fmt::format("Invalid physical row group offsets: {}", prg.ToString()));
    }

    // Count deletions within this physical row group range
    auto lo =
        std::lower_bound(sorted_deletions.begin(), sorted_deletions.end(), static_cast<int64_t>(prg.start_offset));
    auto hi = std::lower_bound(sorted_deletions.begin(), sorted_deletions.end(), static_cast<int64_t>(prg.end_offset));
    auto deletions_in_rg = static_cast<uint64_t>(std::distance(lo, hi));
    auto physical_rows = static_cast<uint64_t>(prg.end_offset - prg.start_offset);
    if (deletions_in_rg > physical_rows) {
      return arrow::Status::Invalid(fmt::format("Invalid deletion count for physical row group: deletions={}, rows={}",
                                                deletions_in_rg, physical_rows));
    }
    uint64_t logical_rows = physical_rows - deletions_in_rg;

    logical_row_group_infos.emplace_back(RowGroupInfo{
        .start_offset = logical_offset,
        .end_offset = logical_offset + logical_rows,
        .memory_size = prg.memory_size,
    });
    logical_offset += logical_rows;
  }

  return logical_row_group_infos;
}

int64_t IcebergFormatReader::logical_to_physical(int64_t logical_offset) const {
  if (!sorted_deletions_ || sorted_deletions_->empty()) {
    return logical_offset;
  }
  int64_t physical = logical_offset;
  while (true) {
    auto num_dels = static_cast<int64_t>(
        std::upper_bound(sorted_deletions_->begin(), sorted_deletions_->end(), physical) - sorted_deletions_->begin());
    int64_t new_physical = logical_offset + num_dels;
    if (new_physical == physical) {
      break;
    }
    physical = new_physical;
  }
  return physical;
}

arrow::Result<std::vector<RowGroupInfo>> IcebergFormatReader::get_row_group_infos() { return logical_row_group_infos_; }

arrow::Result<std::shared_ptr<arrow::RecordBatch>> IcebergFormatReader::get_chunk(const int& row_group_index) {
  // Check if this logical row group has zero rows (e.g. all rows deleted)
  if (row_group_index >= 0 && static_cast<size_t>(row_group_index) < logical_row_group_infos_.size()) {
    const auto& lrg = logical_row_group_infos_[row_group_index];
    if (lrg.start_offset == lrg.end_offset) {
      return arrow::RecordBatch::MakeEmpty(output_schema());
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto batch, inner_reader_->get_chunk(row_group_index));

  if (!deleted_positions_ || deleted_positions_->empty()) {
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

  if (!deleted_positions_ || deleted_positions_->empty()) {
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
  if (row_indices.empty()) {
    return arrow::Table::MakeEmpty(output_schema());
  }

  if (!deleted_positions_ || deleted_positions_->empty()) {
    return inner_reader_->take(row_indices);
  }

  // Map logical indices (post-delete) to physical indices (pre-delete)
  // using the pre-sorted sorted_deletions_ member.
  std::vector<int64_t> physical;
  physical.reserve(row_indices.size());
  for (auto logical_idx : row_indices) {
    physical.push_back(logical_to_physical(logical_idx));
  }

  return inner_reader_->take(physical);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> IcebergFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  // Empty range — return immediately (e.g. all rows deleted)
  if (start_offset >= end_offset) {
    ARROW_ASSIGN_OR_RAISE(auto empty_batch, arrow::RecordBatch::MakeEmpty(output_schema()));
    return arrow::RecordBatchReader::Make({empty_batch});
  }

  if (!deleted_positions_ || deleted_positions_->empty()) {
    return inner_reader_->read_with_range(start_offset, end_offset);
  }

  // Map logical range to physical range
  auto physical_start = static_cast<uint64_t>(logical_to_physical(static_cast<int64_t>(start_offset)));
  auto physical_end = (end_offset > start_offset)
                          ? static_cast<uint64_t>(logical_to_physical(static_cast<int64_t>(end_offset) - 1) + 1)
                          : physical_start;

  // Read physical range, then filter deleted rows
  ARROW_ASSIGN_OR_RAISE(auto inner_rbreader, inner_reader_->read_with_range(physical_start, physical_end));
  ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatchReader(inner_rbreader.get()));
  ARROW_ASSIGN_OR_RAISE(auto batch, table->CombineChunksToBatch());

  ARROW_ASSIGN_OR_RAISE(auto filtered, filter_batch(batch, physical_start));
  return arrow::RecordBatchReader::Make({filtered});
}

std::shared_ptr<arrow::Schema> IcebergFormatReader::get_schema() const { return inner_reader_->get_schema(); }

arrow::Result<std::shared_ptr<FormatReader>> IcebergFormatReader::clone_reader() {
  ARROW_ASSIGN_OR_RAISE(auto cloned_inner, inner_reader_->clone_reader());
  auto cloned_parquet = std::dynamic_pointer_cast<parquet::ParquetFormatReader>(cloned_inner);
  if (!cloned_parquet) {
    return arrow::Status::Invalid("Failed to clone inner ParquetFormatReader");
  }

  return std::shared_ptr<FormatReader>(new IcebergFormatReader(
      std::move(cloned_parquet), data_file_uri_, properties_, delete_metadata_, deleted_positions_, sorted_deletions_,
      logical_row_group_infos_, needed_columns_, projected_schema_));
}

}  // namespace milvus_storage::iceberg
