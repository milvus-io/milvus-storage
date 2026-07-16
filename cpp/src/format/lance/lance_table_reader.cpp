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

#include "milvus-storage/format/lance/lance_table_reader.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/lance/lance_common.h"

namespace milvus_storage::lance {

LanceTableReader::LanceTableReader(const std::shared_ptr<BlockingDataset>& dataset,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const milvus_storage::api::Properties& properties,
                                   const std::vector<std::string>& needed_columns)
    : dataset_(dataset),
      fragment_id_(fragment_id),
      read_schema_(schema),
      properties_(properties),
      needed_columns_(needed_columns),
      fragment_reader_(nullptr) {}

LanceTableReader::LanceTableReader(const std::string& uri,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const milvus_storage::api::Properties& properties,
                                   const std::vector<std::string>& needed_columns)
    : uri_(uri),
      fragment_id_(fragment_id),
      read_schema_(schema),
      properties_(properties),
      needed_columns_(needed_columns),
      fragment_reader_(nullptr) {}

static arrow::Result<std::vector<uint64_t>> estimate_fragment_column_memory_sizes(const BlockingDataset& dataset,
                                                                                  uint64_t fragment_id,
                                                                                  size_t num_columns) {
  ARROW_ASSIGN_OR_RAISE(auto memory_sizes, dataset.EstimateFragmentColumnMemory(fragment_id));
  if (memory_sizes.size() != num_columns) {
    return arrow::Status::Invalid("Lance column memory estimate count does not match the file schema: ",
                                  memory_sizes.size(), " != ", num_columns);
  }

  uint64_t total_size = 0;
  for (auto memory_size : memory_sizes) {
    if (memory_size > std::numeric_limits<uint64_t>::max() - total_size) {
      return arrow::Status::Invalid("Lance column memory estimates exceed the uint64_t range");
    }
    total_size += memory_size;
  }
  return memory_sizes;
}

static arrow::Result<std::vector<RowGroupInfo>> create_row_group_infos(
    uint64_t rows_in_file, uint64_t logical_chunk_rows, const std::vector<uint64_t>& fragment_column_memory_sizes) {
  if (rows_in_file == 0) {
    return std::vector<RowGroupInfo>{};
  }
  assert(logical_chunk_rows > 0);

  uint64_t fragment_memory_size = 0;
  for (auto column_memory_size : fragment_column_memory_sizes) {
    fragment_memory_size += column_memory_size;
  }

  std::vector<RowGroupInfo> result;
  uint64_t last_offset = 0;
  uint64_t last_memory_offset = 0;

  while (last_offset < rows_in_file) {
    uint64_t end_offset = std::min(last_offset + logical_chunk_rows, rows_in_file);
    // end_offset <= rows_in_file, so the quotient is at most fragment_memory_size and is safe to cast to uint64_t.
    auto memory_offset =
        static_cast<uint64_t>((static_cast<unsigned __int128>(fragment_memory_size) * end_offset) / rows_in_file);
    auto memory_size = memory_offset - last_memory_offset;
    ARROW_ASSIGN_OR_RAISE(auto column_memory_sizes, DistributeMemorySizes(memory_size, fragment_column_memory_sizes));
    result.emplace_back(RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = end_offset,
        .memory_size = memory_size,
        .column_memory_sizes = std::move(column_memory_sizes),
    });
    last_offset = end_offset;
    last_memory_offset = memory_offset;
  }

  return result;
}

static arrow::Result<std::shared_ptr<arrow::Schema>> build_read_schema(
    const std::shared_ptr<arrow::Schema>& file_schema,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns) {
  if (read_schema) {
    return read_schema;
  }
  if (!file_schema) {
    return arrow::Status::Invalid("Lance file schema is not available");
  }
  if (needed_columns.empty()) {
    return file_schema;
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& col : needed_columns) {
    auto field = file_schema->GetFieldByName(col);
    if (!field) {
      return arrow::Status::Invalid(
          fmt::format("Lance column '{}' not found in fragment schema: {}", col, file_schema->ToString()));
    }
    fields.push_back(field);
  }
  return arrow::schema(fields);
}

std::string LanceTableReader::MetaTrait::cache_key(const milvus_storage::api::ColumnGroupFile& file) {
  auto cache_key = fmt::format("lance-table|path:{}|file_size:{}|footer_size:{}", file.path,
                               file.Get<uint64_t>(milvus_storage::api::kPropertyFileSize),
                               file.Get<uint64_t>(milvus_storage::api::kPropertyFooterSize));
  auto metadata_it = file.properties.find(milvus_storage::api::kPropertyMetadata);
  if (metadata_it != file.properties.end()) {
    cache_key += "|metadata:" + metadata_it->second;
  }
  return cache_key;
}

arrow::Result<LanceTableReader::MetaTrait::MetadataPtr> LanceTableReader::MetaTrait::load_metadata(
    const milvus_storage::api::ColumnGroupFile& file,
    const milvus_storage::api::Properties& properties,
    const KeyRetriever& key_retriever) {
  (void)key_retriever;

  ARROW_ASSIGN_OR_RAISE(auto parsed_uri, ParseLanceUri(file.path));
  auto base_uri = std::move(parsed_uri.first);
  auto fragment_id = parsed_uri.second;

  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, base_uri));
  auto lance_uri = ToStandardLanceUri(base_uri);

  std::shared_ptr<BlockingDataset> dataset;
  try {
    dataset = BlockingDataset::Open(lance_uri, ToStorageOptions(fs_config));
  } catch (const std::exception& e) {
    return arrow::Status::IOError("Failed to open Lance dataset for metadata: ", e.what());
  }

  std::shared_ptr<arrow::Schema> file_schema;
  {
    ArrowSchema c_fragment_schema;
    try {
      dataset->GetFragmentSchema(fragment_id, c_fragment_schema);
    } catch (const LanceException& e) {
      return arrow::Status::IOError(fmt::format("Failed to get fragment schema: {}", e.what()));
    }
    ARROW_ASSIGN_OR_RAISE(file_schema, arrow::ImportSchema(&c_fragment_schema));
  }

  uint64_t logical_rows = 0;
  uint64_t physical_rows = 0;
  try {
    logical_rows = dataset->GetFragmentRowCount(fragment_id);
    physical_rows = dataset->GetFragmentPhysicalRowCount(fragment_id);
  } catch (const LanceException& e) {
    return arrow::Status::IOError("Failed to get row counts for Lance fragment ", fragment_id, ": ", e.what());
  }
  if (physical_rows < logical_rows) {
    return arrow::Status::Invalid("Fragment ", fragment_id, " has inconsistent metadata: physical_rows (",
                                  physical_rows, ") < logical_rows (", logical_rows, ")");
  }

  ARROW_ASSIGN_OR_RAISE(auto logical_chunk_rows,
                        milvus_storage::api::GetValue<uint64_t>(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS));

  ARROW_ASSIGN_OR_RAISE(
      auto fragment_column_memory_sizes,
      estimate_fragment_column_memory_sizes(*dataset, fragment_id, static_cast<size_t>(file_schema->num_fields())));
  ARROW_ASSIGN_OR_RAISE(auto row_group_infos,
                        create_row_group_infos(logical_rows, logical_chunk_rows, fragment_column_memory_sizes));

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key(file);
  metadata->path = file.path;
  metadata->file_schema = std::move(file_schema);
  metadata->row_group_infos = std::move(row_group_infos);
  size_t column_memory_sizes_size = 0;
  for (const auto& row_group_info : metadata->row_group_infos) {
    column_memory_sizes_size += row_group_info.column_memory_sizes.size() * sizeof(uint64_t);
  }
  auto outer_metadata_size =
      sizeof(Metadata) + metadata->row_group_infos.size() * sizeof(RowGroupInfo) + column_memory_sizes_size;
  metadata->cache_size = outer_metadata_size + file.Get<uint64_t>(milvus_storage::api::kPropertyFooterSize);
  metadata->payload = Payload{
      .base_uri = std::move(base_uri),
      .fragment_id = fragment_id,
      .dataset = std::move(dataset),
      .logical_row_count = logical_rows,
      .physical_row_count = physical_rows,
      .num_deletions = physical_rows - logical_rows,
      .logical_chunk_rows = logical_chunk_rows,
      .properties = properties,
  };

  MetadataPtr result = metadata;
  return result;
}

arrow::Result<std::shared_ptr<LanceTableReader>> LanceTableReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const milvus_storage::api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  (void)file;
  (void)predicate;
  if (!metadata) {
    return arrow::Status::Invalid("Cannot open Lance reader from null metadata");
  }
  if (!metadata->payload.dataset) {
    return arrow::Status::Invalid("Cannot open Lance reader from metadata with null dataset");
  }

  auto reader = std::make_shared<LanceTableReader>(metadata->payload.dataset, metadata->payload.fragment_id,
                                                   read_schema, metadata->payload.properties, needed_columns);
  reader->uri_ = metadata->payload.base_uri;
  reader->file_schema_ = metadata->file_schema;
  reader->logical_chunk_rows_ = metadata->payload.logical_chunk_rows;
  reader->num_deletions_ = metadata->payload.num_deletions;
  reader->row_group_infos_ = metadata->row_group_infos;

  ARROW_ASSIGN_OR_RAISE(auto requested_schema, build_read_schema(reader->file_schema_, read_schema, needed_columns));
  ArrowSchema c_arrow_schema;
  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*requested_schema, &c_arrow_schema));

  try {
    reader->fragment_reader_ =
        BlockingFragmentReader::Open(*metadata->payload.dataset, metadata->payload.fragment_id, c_arrow_schema);
  } catch (const LanceException& e) {
    if (c_arrow_schema.release) {
      c_arrow_schema.release(&c_arrow_schema);
    }
    return arrow::Status::IOError("Failed to open Lance fragment reader for fragment ", metadata->payload.fragment_id,
                                  ": ", e.what());
  }

  return reader;
}

arrow::Status LanceTableReader::open() {
  assert(!fragment_reader_);

  if (!dataset_) {
    // uri_ is in Milvus format (scheme://address/bucket/key) so extfs.<alias>.*
    // can be resolved by address+bucket. Strip the address back to standard form
    // (scheme://bucket/key) before handing to Lance, whose object_store treats
    // the host as the bucket.
    ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties_, uri_));
    auto lance_uri = ToStandardLanceUri(uri_);
    LOG_STORAGE_DEBUG_ << "uri=" << uri_ << ", lance_uri=" << lance_uri << ", alias=" << fs_config.alias
                       << ", role_arn=" << (fs_config.role_arn.empty() ? "(empty)" : fs_config.role_arn)
                       << ", external_id_set=" << (fs_config.external_id.empty() ? "false" : "true")
                       << ", use_iam=" << fs_config.use_iam;
    dataset_ = BlockingDataset::Open(lance_uri, ToStorageOptions(fs_config));
  }

  // Always derive file schema from fragment metadata
  {
    ArrowSchema c_fragment_schema;
    try {
      dataset_->GetFragmentSchema(fragment_id_, c_fragment_schema);
    } catch (const LanceException& e) {
      return arrow::Status::IOError(fmt::format("Failed to get fragment schema: {}", e.what()));
    }
    ARROW_ASSIGN_OR_RAISE(file_schema_, arrow::ImportSchema(&c_fragment_schema));
  }

  // Build the read schema for fragment reader:
  // use user-provided schema if available, otherwise project file schema by needed_columns
  ARROW_ASSIGN_OR_RAISE(auto read_schema, build_read_schema(file_schema_, read_schema_, needed_columns_));

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));

  ARROW_ASSIGN_OR_RAISE(
      auto fragment_column_memory_sizes,
      estimate_fragment_column_memory_sizes(*dataset_, fragment_id_, static_cast<size_t>(file_schema_->num_fields())));

  ArrowSchema c_arrow_schema;
  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*read_schema, &c_arrow_schema));

  fragment_reader_ = BlockingFragmentReader::Open(*dataset_, fragment_id_, c_arrow_schema);

  // Lance's read_range accepts logical indices (post-deletion) and internally
  // patches the range to skip deleted rows. So row_group_infos uses logical row count.
  // However, read_range's batch_size is applied to the *physical* range after
  // patch_range_for_deletions, so we add num_deletions_ to batch_size to ensure
  // each read produces a single output batch.
  auto logical_rows = fragment_reader_->RowCount();
  try {
    auto physical_rows = dataset_->GetFragmentPhysicalRowCount(fragment_id_);
    if (physical_rows < logical_rows) {
      return arrow::Status::Invalid("Fragment ", fragment_id_, " has inconsistent metadata: physical_rows (",
                                    physical_rows, ") < logical_rows (", logical_rows, ")");
    }
    num_deletions_ = physical_rows - logical_rows;
  } catch (const lance::LanceException& e) {
    return arrow::Status::IOError("Failed to get physical row count for fragment ", fragment_id_, ": ", e.what());
  }
  ARROW_ASSIGN_OR_RAISE(row_group_infos_,
                        create_row_group_infos(logical_rows, logical_chunk_rows_, fragment_column_memory_sizes));

  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> LanceTableReader::get_schema() const { return file_schema_; }

arrow::Result<std::vector<RowGroupInfo>> LanceTableReader::get_row_group_infos() {
  assert(fragment_reader_);
  return row_group_infos_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> LanceTableReader::get_chunk(const int& row_group_index) {
  assert(fragment_reader_);
  auto start_idx = row_group_infos_[row_group_index].start_offset;
  auto end_idx = row_group_infos_[row_group_index].end_offset;
  // FIXME: Lance's read_range may produce multiple output batches for two reasons:
  // 1. batch_size is applied to the *physical* range (after patch_range_for_deletions),
  //    so deletions cause the physical range to exceed batch_size.
  // 2. Lance may split at internal page boundaries regardless of batch_size.
  // We add num_deletions_ to mitigate (1), but (2) is not addressed — if Lance
  // splits at page boundaries, chunk(0) will silently lose trailing rows in Release
  // builds (assert is a no-op). A robust fix would combine all chunks here.
  ArrowArrayStream array_stream =
      fragment_reader_->ReadRangesAsStream(start_idx, end_idx, end_idx - start_idx + num_deletions_);
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);
  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> LanceTableReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  assert(fragment_reader_);
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;

#ifndef NDEBUG
  // verify rg_indices_in_file have been sorted
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    assert(rg_indices_in_file[i] >= rg_indices_in_file[i - 1]);
  }
#endif

  std::vector<std::pair<uint64_t, uint64_t>> rg_idx_ranges;

  // calc continuous ranges
  // ex. [1, 2, 3, 5] -> [(1, 3), (5, 5)]
  size_t start_idx = 0;
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    if (rg_indices_in_file[i] != rg_indices_in_file[i - 1] + 1) {
      rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file[i - 1]);
      start_idx = i;
    }
  }

  if (start_idx < rg_indices_in_file.size()) {
    rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file.back());
  }

  for (const auto& rg_range : rg_idx_ranges) {
    // load continuous chunks in one read
    const auto& start_rg_info = row_group_infos_[rg_range.first];
    const auto& end_rg_info = row_group_infos_[rg_range.second];

    // batch_size adds num_deletions_ for the same reason as get_chunk — see comment there.
    ArrowArrayStream array_stream =
        fragment_reader_->ReadRangesAsStream(start_rg_info.start_offset, end_rg_info.end_offset,
                                             end_rg_info.end_offset - start_rg_info.start_offset + num_deletions_);
    ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
    assert(chunkedarray != nullptr);

    // assign to rbs
    for (size_t j = 0; j < chunkedarray->num_chunks(); ++j) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(j)));
      rbs.emplace_back(rb);
    }
  }

  return rbs;
}

arrow::Result<std::shared_ptr<arrow::Table>> LanceTableReader::take(const std::vector<int64_t>& row_indices) {
  assert(fragment_reader_);
  ArrowArrayStream array_stream = fragment_reader_->TakeAsStream(row_indices, row_indices.size());
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));

  // out of range
  if (chunkedarray->num_chunks() == 0) {
    return arrow::Status::Invalid(fmt::format("out of row range [0, {}]", fragment_reader_->RowCount()));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  for (size_t i = 0; i < chunkedarray->num_chunks(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
    rbs.emplace_back(rb);
  }

  return arrow::Table::FromRecordBatches(rbs);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> LanceTableReader::read_with_range(const uint64_t& start_offset,
                                                                                           const uint64_t& end_offset) {
  assert(fragment_reader_);
  // Lance's read_range accepts logical indices directly.
  // batch_size adds num_deletions_ for the same reason as get_chunk — see comment there.
  ArrowArrayStream array_stream =
      fragment_reader_->ReadRangesAsStream(start_offset, end_offset, end_offset - start_offset + num_deletions_);
  return arrow::ImportRecordBatchReader(&array_stream);
}

arrow::Result<std::shared_ptr<FormatReader>> LanceTableReader::clone_reader() {
  assert(fragment_reader_);  // already opened
  return this->shared_from_this();
}

}  // namespace milvus_storage::lance
