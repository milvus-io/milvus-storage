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

#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "vortex_bridge.h"

#include <string>
#include <iostream>
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

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::vortex {

namespace {

static inline expr::Expr build_projection(const std::vector<std::string>& ncs) {
  return expr::select(std::vector<std::string_view>(ncs.begin(), ncs.end()), expr::root());
}

static void remove_metadata_from_schema(ArrowSchema* schema) {
  assert(schema != nullptr);
  for (int64_t i = 0; i < schema->n_children; ++i) {
    remove_metadata_from_schema(schema->children[i]);
  }
  schema->metadata = nullptr;
}

static inline arrow::Result<ArrowSchema> export_c_arrow_schema(const std::shared_ptr<arrow::Schema>& schema) {
  ArrowSchema c_arrow_schema;

  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*schema, &c_arrow_schema));
  // Can't use metadata in vortex, because different implementation of arrow in rust and c++:
  // the datatype compare in rust will check the metadata equality, but vortex won't record the metadata
  // in writer side. So caller have to set metadata to nullptr here.
  // Don't consider memory leak here, direct set to nullptr is safe, because the metadata alloced by arrow::ExportSchema
  // which stored in the private_data of c_schema_, so when c_schema_ destructed, the metadata will be freed too.
  remove_metadata_from_schema(&c_arrow_schema);

  return std::move(c_arrow_schema);
}

static std::shared_ptr<VortexFile> open_shared_vortex_file(const std::shared_ptr<FileSystemWrapper>& fs_holder,
                                                           const std::string& path,
                                                           uint64_t file_size,
                                                           uint64_t footer_size) {
  auto vxfile = VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder.get()), path, file_size, footer_size);
  return std::shared_ptr<VortexFile>(std::move(vxfile));
}

static arrow::Result<std::shared_ptr<arrow::Schema>> import_vortex_file_schema(const VortexFile& vxfile) {
  ArrowSchema c_schema;
  try {
    vxfile.GetFileSchema(c_schema);
  } catch (const VortexException& e) {
    return arrow::Status::IOError(fmt::format("Failed to get vortex file schema: {}", e.what()));
  }
  return arrow::ImportSchema(&c_schema);
}

static std::vector<RowGroupInfo> create_row_group_infos(uint64_t memory_usage_in_file,
                                                        uint64_t rows_in_file,
                                                        const std::vector<uint64_t>& row_ranges) {
  if (rows_in_file == 0) {
    return {};
  }

  std::vector<RowGroupInfo> result(row_ranges.size());
  uint64_t last_offset = 0;

  for (size_t i = 0; i < row_ranges.size(); i++) {
    result[i] = RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = last_offset + row_ranges[i],
        .memory_size = memory_usage_in_file * (row_ranges[i] / rows_in_file),
    };
    last_offset += row_ranges[i];
  }
  return result;
}

static arrow::Result<std::shared_ptr<arrow::Schema>> projected_file_schema(
    const std::shared_ptr<arrow::Schema>& file_schema,
    const std::vector<std::string>& projected_columns,
    const std::string& path) {
  if (!file_schema) {
    return arrow::Status::Invalid(fmt::format("Vortex file schema is not initialized. [path={}]", path));
  }
  if (projected_columns.empty()) {
    return file_schema;
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(projected_columns.size());
  for (const auto& col_name : projected_columns) {
    auto field = file_schema->GetFieldByName(col_name);
    if (!field) {
      return arrow::Status::Invalid(
          fmt::format("Column '{}' not found in vortex file schema. [path={}]", col_name, path));
    }
    fields.emplace_back(std::move(field));
  }
  return arrow::schema(std::move(fields));
}

static arrow::Result<std::shared_ptr<arrow::Schema>> output_schema_for_empty_batch(
    const std::shared_ptr<arrow::Schema>& file_schema,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& projected_columns,
    const std::string& path) {
  if (read_schema) {
    return read_schema;
  }
  return projected_file_schema(file_schema, projected_columns, path);
}

static std::vector<uint64_t> recalc_row_ranges(const std::vector<uint64_t>& original_ranges,
                                               size_t logical_chunk_rows) {
  std::vector<uint64_t> new_ranges;
  for (const auto& range : original_ranges) {
    size_t full_chunks = range / logical_chunk_rows;
    size_t remainder = range % logical_chunk_rows;

    for (size_t i = 0; i < full_chunks; i++) {
      new_ranges.emplace_back(logical_chunk_rows);
    }

    if (remainder > 0) {
      new_ranges.emplace_back(remainder);
    }
  }

  return new_ranges;
}

static uint64_t compute_total_mem_usage(const VortexFile& vxfile) {
  auto sizes = vxfile.GetUncompressedSizes();
  if (sizes.empty()) {
    return 0;
  }
  uint64_t total_size = 0;
  for (auto size : sizes) {
    if (size == UINT64_MAX) {
      return 0;
    }
    total_size += size;
  }
  return total_size;
}

}  // namespace

std::string VortexFormatReader::MetaTrait::cache_key(const api::ColumnGroupFile& file) {
  std::string key =
      fmt::format("vortex:path={};file_size={};footer_size={}", file.path, file.Get<uint64_t>(api::kPropertyFileSize),
                  file.Get<uint64_t>(api::kPropertyFooterSize));
  auto metadata = file.properties.find(api::kPropertyMetadata);
  if (metadata != file.properties.end()) {
    key += fmt::format(";metadata_size={};metadata={}", metadata->second.size(), metadata->second);
  }
  return key;
}

arrow::Result<VortexFormatReader::MetaTrait::MetadataPtr> VortexFormatReader::MetaTrait::load_metadata(
    const api::ColumnGroupFile& file, const api::Properties& properties, const KeyRetriever& key_retriever) {
  (void)key_retriever;

  auto key = cache_key(file);
  const auto file_size = file.Get<uint64_t>(api::kPropertyFileSize);
  const auto footer_size = file.Get<uint64_t>(api::kPropertyFooterSize);

  ARROW_ASSIGN_OR_RAISE(auto logical_chunk_rows,
                        api::GetValue<uint64_t>(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, file.path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(file.path));

  auto fs_holder = std::make_shared<FileSystemWrapper>(fs);
  auto vxfile = open_shared_vortex_file(fs_holder, uri.key, file_size, footer_size);

  ARROW_ASSIGN_OR_RAISE(auto file_schema, import_vortex_file_schema(*vxfile));

  const auto row_ranges = vxfile->Splits();
  const auto memory_usage = compute_total_mem_usage(*vxfile);
  auto row_group_infos =
      create_row_group_infos(memory_usage, vxfile->RowCount(), recalc_row_ranges(row_ranges, logical_chunk_rows));

  auto metadata = std::make_shared<Metadata>(Metadata{
      .cache_key = std::move(key),
      .path = uri.key,
      .file_schema = std::move(file_schema),
      .row_group_infos = std::move(row_group_infos),
      .cache_size = footer_size,
      .payload =
          Payload{
              .fs_holder = std::move(fs_holder),
              .vxfile = std::move(vxfile),
              .logical_chunk_rows = logical_chunk_rows,
              .properties = properties,
          },
  });
  return std::static_pointer_cast<const Metadata>(metadata);
}

arrow::Result<std::shared_ptr<VortexFormatReader>> VortexFormatReader::MetaTrait::create_from_metadata(
    MetadataPtr metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  (void)predicate;
  if (!metadata || !metadata->payload.vxfile || !metadata->payload.fs_holder) {
    return arrow::Status::Invalid("Cannot open vortex reader from incomplete metadata");
  }

  auto reader = std::shared_ptr<VortexFormatReader>(
      new VortexFormatReader(metadata, file.Get<uint64_t>(api::kPropertyFileSize),
                             file.Get<uint64_t>(api::kPropertyFooterSize), read_schema, needed_columns));
  return reader;
}

VortexFormatReader::VortexFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                       const std::shared_ptr<arrow::Schema>& schema,
                                       const std::string& path,
                                       const milvus_storage::api::Properties& properties,
                                       const std::vector<std::string>& needed_columns,
                                       uint64_t file_size,
                                       uint64_t footer_size)
    : fs_holder_(std::make_shared<FileSystemWrapper>(fs)),
      proj_cols_(needed_columns),
      path_(path),
      read_schema_(schema),
      properties_(properties),
      file_size_(file_size),
      footer_size_(footer_size),
      vxfile_(nullptr) {}

VortexFormatReader::VortexFormatReader(MetaTrait::MetadataPtr metadata,
                                       uint64_t file_size,
                                       uint64_t footer_size,
                                       const std::shared_ptr<arrow::Schema>& read_schema,
                                       const std::vector<std::string>& needed_columns)
    : fs_holder_(metadata->payload.fs_holder),
      proj_cols_(needed_columns),
      path_(metadata->path),
      read_schema_(read_schema),
      properties_(metadata->payload.properties),
      file_size_(file_size),
      footer_size_(footer_size),
      file_schema_(metadata->file_schema),
      logical_chunk_rows_(metadata->payload.logical_chunk_rows),
      row_group_infos_(metadata->row_group_infos),
      vxfile_(metadata->payload.vxfile) {
  if (read_schema_ && read_schema_->num_fields() == 0) {
    read_schema_ = nullptr;
  }
}

arrow::Status VortexFormatReader::open() {
  assert(!vxfile_);

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  if (read_schema_ && read_schema_->num_fields() == 0) {
    read_schema_ = nullptr;
  }
  vxfile_ = open_shared_vortex_file(fs_holder_, path_, file_size_, footer_size_);

  // Always derive full file schema from file metadata
  ARROW_ASSIGN_OR_RAISE(file_schema_, import_vortex_file_schema(*vxfile_));

  row_group_infos_ =
      create_row_group_infos(total_mem_usage(), rows(), recalc_row_ranges(row_ranges(), logical_chunk_rows_));

  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> VortexFormatReader::get_schema() const { return file_schema_; }

arrow::Result<std::vector<RowGroupInfo>> VortexFormatReader::get_row_group_infos() {
  assert(vxfile_);
  return row_group_infos_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::get_chunk(const int& row_group_index) {
  assert(vxfile_);
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, blocking_read(row_group_infos_[row_group_index].start_offset,
                                                         row_group_infos_[row_group_index].end_offset));
  assert(chunkedarray != nullptr);

  if (chunkedarray->num_chunks() == 0) {
    ARROW_ASSIGN_OR_RAISE(auto output_schema,
                          output_schema_for_empty_batch(file_schema_, read_schema_, proj_cols_, path_));
    return arrow::RecordBatch::MakeEmpty(output_schema);
  }

  assert(chunkedarray->num_chunks() == 1);
  ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0)));
  return rb;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> VortexFormatReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  assert(vxfile_);
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

    ARROW_ASSIGN_OR_RAISE(auto chunked_array, blocking_read(start_rg_info.start_offset, end_rg_info.end_offset));
    // assign to rbs
    for (size_t j = 0; j < chunked_array->num_chunks(); ++j) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunked_array->chunk(j)));
      rbs.emplace_back(rb);
    }
  }

  return rbs;
}

arrow::Result<std::shared_ptr<FormatReader>> VortexFormatReader::clone_reader() {
  assert(vxfile_);  // already opened
  return this->shared_from_this();
}

arrow::Result<ArrowArrayStream> VortexFormatReader::read(uint64_t row_start, uint64_t row_end) {
  auto scan_builder = vxfile_->CreateScanBuilder();
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    scan_builder.WithOutputSchema(c_arrow_schema);
  }

  scan_builder.WithRowRange(row_start, row_end);
  return std::move(scan_builder).IntoStream();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> VortexFormatReader::streaming_read(uint64_t row_start,
                                                                                            uint64_t row_end) {
  ARROW_ASSIGN_OR_RAISE(auto array_stream, read(row_start, row_end));
  return arrow::ImportRecordBatchReader(&array_stream);
}

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> VortexFormatReader::blocking_read(uint64_t row_start,
                                                                                      uint64_t row_end) {
  ARROW_ASSIGN_OR_RAISE(auto array_stream, read(row_start, row_end));
  return arrow::ImportChunkedArray(&array_stream);
}

arrow::Result<std::shared_ptr<arrow::Table>> VortexFormatReader::take(const std::vector<int64_t>& row_indices) {
  assert(vxfile_);
  auto scan_builder = vxfile_->CreateScanBuilder();
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    scan_builder.WithOutputSchema(c_arrow_schema);
  }

  scan_builder.WithIncludeByIndex(reinterpret_cast<const uint64_t*>(row_indices.data()), row_indices.size());

  ArrowArrayStream array_stream = std::move(scan_builder).IntoStream();
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));

  // out of range
  if (chunkedarray->num_chunks() == 0) {
    return arrow::Status::Invalid(fmt::format("out of row range[0, {}].", vxfile_->RowCount()));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  for (size_t i = 0; i < chunkedarray->num_chunks(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
    rbs.emplace_back(rb);
  }

  return arrow::Table::FromRecordBatches(rbs);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> VortexFormatReader::read_with_range(
    const uint64_t& start_offset, const uint64_t& end_offset) {
  assert(vxfile_);
  return streaming_read(start_offset, end_offset);
}

uint64_t VortexFormatReader::total_mem_usage() { return compute_total_mem_usage(*vxfile_); }

}  // namespace milvus_storage::vortex
