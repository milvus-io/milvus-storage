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

#include "milvus-storage/format/vortex/vortex_planner.h"
#include "vortex_bridge.h"

#include <optional>
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

#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::vortex {

extern const ffi::CoalescingWindow kSmallCoalescingWindow{1024 * 1024, 1024 * 1024};

namespace {
const ffi::CoalescingWindow kLargeCoalescingWindow{1024 * 1024, 8 * 1024 * 1024};

struct MemorySizeEstimate {
  uint64_t total_size;
  std::vector<uint64_t> column_sizes;
};

static uint8_t arrow_type_to_tag(const arrow::DataType& dt) {
  switch (dt.id()) {
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
      return 0;  // Int
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
      return 1;  // UInt
    case arrow::Type::HALF_FLOAT:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      return 2;  // Float
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::STRING_VIEW:
      return 3;  // Utf8 — Vortex round-trips arrow::utf8() back as STRING_VIEW
    case arrow::Type::BOOL:
      return 4;  // Bool
    default:
      return 5;  // Other
  }
}

static expr::Expr build_projection(const std::vector<std::string>& ncs) {
  std::vector<std::string_view> field_views;
  field_views.reserve(ncs.size());
  for (const auto& field : ncs) {
    field_views.emplace_back(field);
  }
  return expr::select(field_views, expr::root());
}

static arrow::Result<std::optional<expr::Expr>> parse_predicate_for_schema(
    const std::string& predicate, const std::shared_ptr<arrow::Schema>& file_schema) {
  if (predicate.empty()) {
    return std::nullopt;
  }
  if (!file_schema) {
    return arrow::Status::Invalid("Vortex predicate requires an opened reader with file schema");
  }

  std::vector<expr::PredicateColumn> schema;
  schema.reserve(file_schema->num_fields());
  for (const auto& field : file_schema->fields()) {
    schema.push_back({field->name(), arrow_type_to_tag(*field->type())});
  }
  auto parsed = expr::parse_predicate(predicate, schema);
  if (!parsed.ok()) {
    return arrow::Status::Invalid(
        fmt::format("Failed to parse Vortex predicate '{}': {}", predicate, parsed.status().message()));
  }
  return std::move(parsed).ValueOrDie();
}

static arrow::Result<std::shared_ptr<arrow::Schema>> project_schema(const std::shared_ptr<arrow::Schema>& schema,
                                                                    const std::vector<std::string>& columns) {
  if (!schema) {
    return arrow::Status::Invalid("Vortex output schema requires an opened reader");
  }
  if (columns.empty()) {
    return schema;
  }

  arrow::FieldVector fields;
  fields.reserve(columns.size());
  for (const auto& column : columns) {
    auto field = schema->GetFieldByName(column);
    if (!field) {
      return arrow::Status::KeyError(fmt::format("Vortex projection field not found: {}", column));
    }
    fields.emplace_back(std::move(field));
  }
  return arrow::schema(std::move(fields));
}

static arrow::Status apply_row_ranges_selection(ScanBuilder* scan_builder,
                                                const std::vector<RowRange>& row_ranges,
                                                uint64_t row_count) {
  std::vector<uint64_t> starts;
  std::vector<uint64_t> ends;
  starts.reserve(row_ranges.size());
  ends.reserve(row_ranges.size());
  uint64_t previous_end = 0;
  bool first_range = true;
  for (const auto& range : row_ranges) {
    if (range.start > range.end || range.end > row_count) {
      return arrow::Status::Invalid(
          fmt::format("Vortex read row range [{}, {}) out of rows {}", range.start, range.end, row_count));
    }
    if (range.start == range.end) {
      continue;
    }
    if (!first_range && range.start < previous_end) {
      return arrow::Status::Invalid("Vortex read row ranges must be sorted and non-overlapping");
    }
    first_range = false;
    previous_end = range.end;
    starts.emplace_back(range.start);
    ends.emplace_back(range.end);
  }
  scan_builder->WithRowRanges(starts.data(), ends.data(), starts.size());
  return arrow::Status::OK();
}

static arrow::Status apply_read_plan_selection(ScanBuilder* scan_builder,
                                               const VortexReadPlan& plan,
                                               uint64_t row_count,
                                               const std::string& path) {
  if (const auto* range_scan = std::get_if<VortexReadPlan::RangeScan>(&plan.op)) {
    ARROW_RETURN_NOT_OK(apply_row_ranges_selection(scan_builder, range_scan->ranges, row_count));
    return arrow::Status::OK();
  }

  if (const auto* take = std::get_if<VortexReadPlan::Take>(&plan.op)) {
    std::vector<uint64_t> include_indices;
    include_indices.reserve(take->row_indices.size());
    for (const auto row_index : take->row_indices) {
      if (row_index < 0 || static_cast<uint64_t>(row_index) >= row_count) {
        return arrow::Status::Invalid(
            fmt::format("Row index out of range: {}. [path={}, valid_range=[0, {}]]", row_index, path, row_count));
      }
      include_indices.emplace_back(static_cast<uint64_t>(row_index));
    }
    scan_builder->WithIncludeByIndex(include_indices.data(), include_indices.size());
    // Take::ranges is used by Milvus to pin/load sparse-file cells. Applying
    // those ranges here would re-run the global include indices inside each
    // range and can make variable-length arrays read out of bounds.
    return arrow::Status::OK();
  }

  return arrow::Status::Invalid("Unsupported Vortex read plan operation");
}

static void remove_metadata_from_schema(ArrowSchema* schema) {
  assert(schema != nullptr);
  for (int64_t i = 0; i < schema->n_children; ++i) {
    remove_metadata_from_schema(schema->children[i]);
  }
  schema->metadata = nullptr;
}

static arrow::Result<ArrowSchema> export_c_arrow_schema(const std::shared_ptr<arrow::Schema>& schema) {
  ArrowSchema c_arrow_schema;

  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*schema, &c_arrow_schema));
  // Can't use metadata in vortex, because different implementation of arrow in rust and c++:
  // the datatype compare in rust will check the metadata equality, but vortex won't record the metadata
  // in writer side. So caller have to set metadata to nullptr here.
  // Don't consider memory leak here, direct set to nullptr is safe, because the metadata alloced by arrow::ExportSchema
  // which stored in the private_data of c_schema_, so when c_schema_ destructed, the metadata will be freed too.
  remove_metadata_from_schema(&c_arrow_schema);

  return c_arrow_schema;
}

static arrow::Result<std::shared_ptr<VortexFile>> open_shared_vortex_file(
    const std::shared_ptr<FileSystemWrapper>& fs_holder,
    const std::string& path,
    uint64_t file_size,
    uint64_t footer_size) {
  auto vxfile = VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder.get()), path, file_size, footer_size);
  if (!vxfile.ok()) {
    return MakeVortexErrorStatus("Failed to open vortex file", vxfile.status());
  }
  return std::shared_ptr<VortexFile>(std::move(vxfile).ValueOrDie());
}

static arrow::Result<std::shared_ptr<arrow::Schema>> import_vortex_file_schema(const VortexFile& vxfile) {
  ArrowSchema c_schema;
  ARROW_RETURN_NOT_OK(MakeVortexErrorStatus("Failed to get vortex file schema", vxfile.GetFileSchema(c_schema)));
  return arrow::ImportSchema(&c_schema);
}

static arrow::Result<std::vector<uint64_t>> get_vortex_splits(const VortexFile& vxfile) {
  auto splits = vxfile.Splits();
  if (!splits.ok()) {
    return MakeVortexErrorStatus("Failed to get vortex splits", splits.status());
  }
  return std::move(splits).ValueOrDie();
}

static arrow::Result<std::vector<RowGroupInfo>> create_row_group_infos(
    uint64_t rows_in_file,
    const std::vector<uint64_t>& row_ranges,
    const std::optional<MemorySizeEstimate>& memory_size_estimate) {
  if (rows_in_file == 0) {
    return std::vector<RowGroupInfo>{};
  }

  std::vector<RowGroupInfo> result;
  result.reserve(row_ranges.size());
  uint64_t last_offset = 0;

  for (auto row_range : row_ranges) {
    if (row_range > rows_in_file) {
      return arrow::Status::Invalid("Vortex row range exceeds the file row count");
    }
    uint64_t memory_size = 0;
    std::vector<uint64_t> column_memory_sizes;
    if (memory_size_estimate) {
      // row_range <= rows_in_file, so the quotient is at most the file memory size and is safe to cast to uint64_t.
      memory_size = static_cast<uint64_t>(static_cast<unsigned __int128>(memory_size_estimate->total_size) * row_range /
                                          rows_in_file);
      ARROW_ASSIGN_OR_RAISE(column_memory_sizes,
                            DistributeMemorySizes(memory_size, memory_size_estimate->column_sizes));
    }
    result.emplace_back(RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = last_offset + row_range,
        .memory_size = memory_size,
        .column_memory_sizes = std::move(column_memory_sizes),
        .memory_size_available = memory_size_estimate.has_value(),
    });
    last_offset += row_range;
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

static arrow::Result<std::vector<uint64_t>> get_column_uncompressed_sizes(const VortexFile& vxfile,
                                                                          size_t column_count) {
  if (vxfile.RowCount() == 0) {
    return std::vector<uint64_t>(column_count, 0);
  }
  FIU_RETURN_ON(FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL,
                arrow::Status::NotImplemented("Injected fault: ", FIUKEY_MEMORY_SIZE_ESTIMATION_FAIL));

  auto sizes = vxfile.GetUncompressedSizes();
  if (sizes.empty()) {
    if (column_count == 0) {
      return sizes;
    }
    return arrow::Status::NotImplemented("Vortex column memory size statistics are not available");
  }
  if (sizes.size() != column_count) {
    return arrow::Status::Invalid(fmt::format(
        "Vortex column memory estimate count does not match the file schema: {} != {}", sizes.size(), column_count));
  }
  for (auto size : sizes) {
    if (size == UINT64_MAX) {
      return arrow::Status::NotImplemented("Vortex column memory size statistics are not available");
    }
  }

  return sizes;
}

static arrow::Result<uint64_t> compute_total_mem_usage(const std::vector<uint64_t>& file_column_uncompressed_sizes) {
  uint64_t total_size = 0;
  for (auto size : file_column_uncompressed_sizes) {
    if (size > std::numeric_limits<uint64_t>::max() - total_size) {
      return arrow::Status::Invalid("Vortex column memory estimates exceed the uint64_t range");
    }
    total_size += size;
  }
  return total_size;
}

static std::optional<MemorySizeEstimate> estimate_memory_sizes(const VortexFile& vxfile,
                                                               size_t column_count,
                                                               const std::string& path) {
  auto column_sizes_result = get_column_uncompressed_sizes(vxfile, column_count);
  if (!column_sizes_result.ok()) {
    // Memory statistics are optional. Do not retain the underlying failure in
    // row-group metadata: estimate APIs return a generic NotImplemented status
    // instead. Keep the detailed reason in the debug log for diagnostics only.
    LOG_STORAGE_DEBUG_ << "Vortex memory estimation is unavailable"
                       << ", path=" << path << ", status=" << column_sizes_result.status().ToString();
    return std::nullopt;
  }

  auto column_sizes = std::move(column_sizes_result).ValueOrDie();
  auto total_size_result = compute_total_mem_usage(column_sizes);
  if (!total_size_result.ok()) {
    LOG_STORAGE_DEBUG_ << "Vortex memory estimation is unavailable"
                       << ", path=" << path << ", status=" << total_size_result.status().ToString();
    return std::nullopt;
  }
  return MemorySizeEstimate{
      .total_size = total_size_result.ValueOrDie(),
      .column_sizes = std::move(column_sizes),
  };
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
  ARROW_ASSIGN_OR_RAISE(auto vxfile, open_shared_vortex_file(fs_holder, uri.key, file_size, footer_size));

  ARROW_ASSIGN_OR_RAISE(auto file_schema, import_vortex_file_schema(*vxfile));

  ARROW_ASSIGN_OR_RAISE(auto row_ranges, get_vortex_splits(*vxfile));
  auto memory_size_estimate = estimate_memory_sizes(*vxfile, static_cast<size_t>(file_schema->num_fields()), uri.key);
  ARROW_ASSIGN_OR_RAISE(auto row_group_infos,
                        create_row_group_infos(vxfile->RowCount(), recalc_row_ranges(row_ranges, logical_chunk_rows),
                                               memory_size_estimate));

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
  if (!metadata || !metadata->payload.vxfile || !metadata->payload.fs_holder) {
    return arrow::Status::Invalid("Cannot open vortex reader from incomplete metadata");
  }

  auto reader = std::shared_ptr<VortexFormatReader>(
      new VortexFormatReader(metadata, file.Get<uint64_t>(api::kPropertyFileSize),
                             file.Get<uint64_t>(api::kPropertyFooterSize), read_schema, needed_columns));
  ARROW_ASSIGN_OR_RAISE(
      auto split_row_indices_mode,
      api::GetValue<std::string>(metadata->payload.properties, PROPERTY_READER_VORTEX_SPLIT_ROW_INDICES));
  ARROW_ASSIGN_OR_RAISE(reader->split_row_indices_,
                        VortexFormatReader::parse_split_row_indices_override(split_row_indices_mode));
  ARROW_RETURN_NOT_OK(reader->set_predicate(predicate));
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

VortexFormatReader::~VortexFormatReader() = default;

arrow::Result<std::optional<bool>> VortexFormatReader::parse_split_row_indices_override(const std::string& mode) {
  if (mode == "auto") {
    return std::nullopt;
  }
  if (mode == "true") {
    return true;
  }
  if (mode == "false") {
    return false;
  }
  return arrow::Status::Invalid(fmt::format("Invalid Vortex split row indices mode: {}", mode));
}

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
  ARROW_ASSIGN_OR_RAISE(auto split_row_indices_mode,
                        api::GetValue<std::string>(properties_, PROPERTY_READER_VORTEX_SPLIT_ROW_INDICES));
  ARROW_ASSIGN_OR_RAISE(split_row_indices_, parse_split_row_indices_override(split_row_indices_mode));
  if (read_schema_ && read_schema_->num_fields() == 0) {
    read_schema_ = nullptr;
  }
  ARROW_ASSIGN_OR_RAISE(vxfile_, open_shared_vortex_file(fs_holder_, path_, file_size_, footer_size_));

  // Always derive full file schema from file metadata
  ARROW_ASSIGN_OR_RAISE(file_schema_, import_vortex_file_schema(*vxfile_));

  ARROW_ASSIGN_OR_RAISE(auto row_ranges, get_vortex_splits(*vxfile_));
  auto memory_size_estimate = estimate_memory_sizes(*vxfile_, static_cast<size_t>(file_schema_->num_fields()), path_);
  ARROW_ASSIGN_OR_RAISE(row_group_infos_,
                        create_row_group_infos(vxfile_->RowCount(), recalc_row_ranges(row_ranges, logical_chunk_rows_),
                                               memory_size_estimate));

  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> VortexFormatReader::get_schema() const { return file_schema_; }

arrow::Status VortexFormatReader::set_predicate(const std::string& predicate) {
  if (predicate.empty()) {
    return arrow::Status::OK();
  }
  if (!file_schema_) {
    return arrow::Status::Invalid("predicate set before file schema is available");
  }
  ARROW_ASSIGN_OR_RAISE(auto maybe_expr, parse_predicate_for_schema(predicate, file_schema_));
  if (maybe_expr.has_value()) {
    parsed_predicate_ = std::make_unique<expr::Expr>(std::move(*maybe_expr));
  }
  return arrow::Status::OK();
}

arrow::Result<std::vector<uint64_t>> VortexFormatReader::row_ranges() const {
  assert(vxfile_);
  return vxfile_->Splits();
}

size_t VortexFormatReader::rows() const {
  assert(vxfile_);
  return vxfile_->RowCount();
}

arrow::Result<std::shared_ptr<arrow::Schema>> VortexFormatReader::output_schema() const {
  if (read_schema_) {
    return read_schema_;
  }
  return project_schema(file_schema_, proj_cols_);
}

arrow::Result<std::vector<RowGroupInfo>> VortexFormatReader::get_row_group_infos() {
  assert(vxfile_);
  return row_group_infos_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::get_chunk(const int& row_group_index) {
  assert(vxfile_);
  if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= row_group_infos_.size()) {
    return arrow::Status::Invalid(
        fmt::format("Vortex row group index {} out of range {}", row_group_index, row_group_infos_.size()));
  }
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray,
                        blocking_read(row_group_infos_[row_group_index].start_offset,
                                      row_group_infos_[row_group_index].end_offset, kLargeCoalescingWindow));
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
  if (rg_indices_in_file.empty()) {
    return rbs;
  }

#ifndef NDEBUG
  // verify rg_indices_in_file have been sorted
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    assert(rg_indices_in_file[i] >= rg_indices_in_file[i - 1]);
  }
#endif

  std::vector<std::pair<uint64_t, uint64_t>> rg_idx_ranges;
  for (const auto row_group_index : rg_indices_in_file) {
    if (row_group_index < 0 || static_cast<size_t>(row_group_index) >= row_group_infos_.size()) {
      return arrow::Status::Invalid(
          fmt::format("Vortex row group index {} out of range {}", row_group_index, row_group_infos_.size()));
    }
  }

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

    ARROW_ASSIGN_OR_RAISE(auto chunked_array,
                          blocking_read(start_rg_info.start_offset, end_rg_info.end_offset, kLargeCoalescingWindow));
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

arrow::Result<ArrowArrayStream> VortexFormatReader::read_with_plan(const VortexReadPlan& plan) {
  if (!vxfile_) {
    return arrow::Status::Invalid("VortexFormatReader is not opened");
  }

  ARROW_ASSIGN_OR_RAISE(auto scan_builder, vxfile_->CreateScanBuilder(kSmallCoalescingWindow));
  if (split_row_indices_.has_value()) {
    scan_builder.WithSplitRowIndices(*split_row_indices_);
  }
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    ARROW_RETURN_NOT_OK(
        MakeVortexErrorStatus("Failed to read vortex file with plan", scan_builder.WithOutputSchema(c_arrow_schema)));
  }

  if (plan.apply_predicate) {
    ARROW_ASSIGN_OR_RAISE(auto parsed_predicate, parse_predicate_for_schema(plan.predicate, file_schema_));
    if (parsed_predicate.has_value()) {
      scan_builder.WithFilter(*parsed_predicate);
    }
  }

  ARROW_RETURN_NOT_OK(apply_read_plan_selection(&scan_builder, plan, vxfile_->RowCount(), path_));

  auto stream = std::move(scan_builder).IntoStream();
  if (!stream.ok()) {
    return MakeVortexErrorStatus("Failed to read vortex file with plan", stream.status());
  }
  return std::move(stream).ValueOrDie();
}

arrow::Result<ArrowArrayStream> VortexFormatReader::read_row_ids_with_plan(const VortexReadPlan& plan) {
  if (!vxfile_) {
    return arrow::Status::Invalid("VortexFormatReader is not opened");
  }

  ARROW_ASSIGN_OR_RAISE(auto scan_builder, vxfile_->CreateScanBuilder(kSmallCoalescingWindow));
  if (split_row_indices_.has_value()) {
    scan_builder.WithSplitRowIndices(*split_row_indices_);
  }
  scan_builder.WithRowIndicesProjection("__milvus_row_id");

  if (plan.apply_predicate) {
    ARROW_ASSIGN_OR_RAISE(auto parsed_predicate, parse_predicate_for_schema(plan.predicate, file_schema_));
    if (parsed_predicate.has_value()) {
      scan_builder.WithFilter(*parsed_predicate);
    }
  }

  ARROW_RETURN_NOT_OK(apply_read_plan_selection(&scan_builder, plan, vxfile_->RowCount(), path_));
  auto stream = std::move(scan_builder).IntoStream();
  if (!stream.ok()) {
    return MakeVortexErrorStatus("Failed to read vortex row ids with plan", stream.status());
  }
  return std::move(stream).ValueOrDie();
}

arrow::Result<ArrowArrayStream> VortexFormatReader::read(uint64_t row_start,
                                                         uint64_t row_end,
                                                         const ffi::CoalescingWindow& coalescing_window) {
  ARROW_ASSIGN_OR_RAISE(auto scan_builder, vxfile_->CreateScanBuilder(coalescing_window));
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    ARROW_RETURN_NOT_OK(
        MakeVortexErrorStatus("Failed to read vortex file", scan_builder.WithOutputSchema(c_arrow_schema)));
  }

  if (parsed_predicate_) {
    scan_builder.WithFilter(*parsed_predicate_);
  }

  scan_builder.WithRowRange(row_start, row_end);
  auto stream = std::move(scan_builder).IntoStream();
  if (!stream.ok()) {
    return MakeVortexErrorStatus("Failed to read vortex file", stream.status());
  }
  return std::move(stream).ValueOrDie();
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> VortexFormatReader::streaming_read(
    uint64_t row_start, uint64_t row_end, const ffi::CoalescingWindow& coalescing_window) {
  ARROW_ASSIGN_OR_RAISE(auto array_stream, read(row_start, row_end, coalescing_window));
  auto reader_result = arrow::ImportRecordBatchReader(&array_stream);
  if (!reader_result.ok()) {
    return MakeVortexErrorStatus("Failed to import vortex record batch reader", reader_result.status());
  }
  return internal::WrapVortexRecordBatchReader(reader_result.ValueOrDie());
}

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> VortexFormatReader::blocking_read(
    uint64_t row_start, uint64_t row_end, const ffi::CoalescingWindow& coalescing_window) {
  ARROW_ASSIGN_OR_RAISE(auto array_stream, read(row_start, row_end, coalescing_window));
  auto chunked_array_result = arrow::ImportChunkedArray(&array_stream);
  if (!chunked_array_result.ok()) {
    return MakeVortexErrorStatus("Failed to import vortex chunked array", chunked_array_result.status());
  }
  return chunked_array_result.ValueOrDie();
}

arrow::Result<std::shared_ptr<arrow::Table>> VortexFormatReader::take(const std::vector<int64_t>& row_indices) {
  assert(vxfile_);
  ARROW_ASSIGN_OR_RAISE(auto scan_builder, vxfile_->CreateScanBuilder(kSmallCoalescingWindow));
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    ARROW_RETURN_NOT_OK(
        MakeVortexErrorStatus("Failed to take from vortex file", scan_builder.WithOutputSchema(c_arrow_schema)));
  }

  scan_builder.WithIncludeByIndex(reinterpret_cast<const uint64_t*>(row_indices.data()), row_indices.size());

  auto array_stream = std::move(scan_builder).IntoStream();
  if (!array_stream.ok()) {
    return MakeVortexErrorStatus("Failed to take from vortex file", array_stream.status());
  }
  auto stream = std::move(array_stream).ValueOrDie();
  auto chunkedarray_result = arrow::ImportChunkedArray(&stream);
  if (!chunkedarray_result.ok()) {
    return MakeVortexErrorStatus("Failed to import vortex take result", chunkedarray_result.status());
  }
  auto chunkedarray = chunkedarray_result.ValueOrDie();

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
  return streaming_read(start_offset, end_offset, kLargeCoalescingWindow);
}

arrow::Result<uint64_t> VortexFormatReader::total_mem_usage() {
  ARROW_ASSIGN_OR_RAISE(auto column_sizes,
                        get_column_uncompressed_sizes(*vxfile_, static_cast<size_t>(file_schema_->num_fields())));
  return compute_total_mem_usage(column_sizes);
}

}  // namespace milvus_storage::vortex
