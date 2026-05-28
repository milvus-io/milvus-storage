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

namespace milvus_storage::vortex {

namespace {
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
  try {
    return expr::parse_predicate(predicate, schema);
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to parse Vortex predicate '{}': {}", predicate, e.what()));
  }
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

static std::vector<RowGroupInfo> create_row_group_infos(uint64_t memory_usage_in_file,
                                                        uint64_t rows_in_file,
                                                        const std::vector<uint64_t>& row_ranges) {
  if (rows_in_file == 0) {
    return {};
  }

  std::vector<RowGroupInfo> result;
  result.reserve(row_ranges.size());
  uint64_t last_offset = 0;

  for (auto row_range : row_ranges) {
    result.emplace_back(RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = last_offset + row_range,
        .memory_size = memory_usage_in_file * row_range / rows_in_file,
    });
    last_offset += row_range;
  }
  return result;
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

}  // namespace

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

arrow::Status VortexFormatReader::open() {
  assert(!vxfile_);

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  ARROW_ASSIGN_OR_RAISE(auto split_row_indices_mode,
                        api::GetValue<std::string>(properties_, PROPERTY_READER_VORTEX_SPLIT_ROW_INDICES));
  ARROW_ASSIGN_OR_RAISE(split_row_indices_, parse_split_row_indices_override(split_row_indices_mode));
  if (read_schema_ && read_schema_->num_fields() == 0) {
    read_schema_ = nullptr;
  }
  vxfile_ = VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder_.get()), path_, file_size_, footer_size_);

  // Always derive full file schema from file metadata
  {
    ArrowSchema c_schema;
    try {
      vxfile_->GetFileSchema(c_schema);
    } catch (const VortexException& e) {
      return arrow::Status::IOError(fmt::format("Failed to get vortex file schema: {}", e.what()));
    }
    ARROW_ASSIGN_OR_RAISE(file_schema_, arrow::ImportSchema(&c_schema));
  }

  row_group_infos_ = create_row_group_infos(total_mem_usage(), vxfile_->RowCount(),
                                            recalc_row_ranges(vxfile_->Splits(), logical_chunk_rows_));

  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> VortexFormatReader::get_schema() const { return file_schema_; }

void VortexFormatReader::set_predicate(const std::string& predicate) {
  if (predicate.empty()) {
    return;
  }
  if (!file_schema_) {
    throw VortexException("predicate set before file schema is available");
  }
  std::vector<expr::PredicateColumn> schema;
  schema.reserve(file_schema_->num_fields());
  for (const auto& field : file_schema_->fields()) {
    schema.push_back({field->name(), arrow_type_to_tag(*field->type())});
  }
  auto maybe_expr = expr::parse_predicate(predicate, schema);
  if (maybe_expr.has_value()) {
    parsed_predicate_ = std::make_unique<expr::Expr>(std::move(*maybe_expr));
  }
}

std::vector<uint64_t> VortexFormatReader::row_ranges() const {
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
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, blocking_read(row_group_infos_[row_group_index].start_offset,
                                                         row_group_infos_[row_group_index].end_offset));
  assert(chunkedarray != nullptr);

  if (chunkedarray->num_chunks() == 0) {
    ARROW_ASSIGN_OR_RAISE(auto schema, output_schema());
    return arrow::RecordBatch::MakeEmpty(schema);
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

arrow::Result<ArrowArrayStream> VortexFormatReader::read_with_plan(const VortexReadPlan& plan) {
  if (!vxfile_) {
    return arrow::Status::Invalid("VortexFormatReader is not opened");
  }

  auto read_file =
      VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder_.get()), path_, file_size_, footer_size_);
  auto scan_builder = read_file->CreateScanBuilder();
  if (split_row_indices_.has_value()) {
    scan_builder.WithSplitRowIndices(*split_row_indices_);
  }
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    scan_builder.WithOutputSchema(c_arrow_schema);
  }

  if (plan.apply_predicate) {
    ARROW_ASSIGN_OR_RAISE(auto parsed_predicate, parse_predicate_for_schema(plan.predicate, file_schema_));
    if (parsed_predicate.has_value()) {
      scan_builder.WithFilter(*parsed_predicate);
    }
  }

  ARROW_RETURN_NOT_OK(apply_read_plan_selection(&scan_builder, plan, read_file->RowCount(), path_));

  return std::move(scan_builder).IntoStream();
}

arrow::Result<ArrowArrayStream> VortexFormatReader::read_row_ids_with_plan(const VortexReadPlan& plan) {
  if (!vxfile_) {
    return arrow::Status::Invalid("VortexFormatReader is not opened");
  }

  auto read_file =
      VortexFile::OpenUnique(reinterpret_cast<uint8_t*>(fs_holder_.get()), path_, file_size_, footer_size_);
  auto scan_builder = read_file->CreateScanBuilder();
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

  ARROW_RETURN_NOT_OK(apply_read_plan_selection(&scan_builder, plan, read_file->RowCount(), path_));
  return std::move(scan_builder).IntoStream();
}

arrow::Result<ArrowArrayStream> VortexFormatReader::read(uint64_t row_start, uint64_t row_end) {
  auto scan_builder = vxfile_->CreateScanBuilder();
  if (split_row_indices_.has_value()) {
    scan_builder.WithSplitRowIndices(*split_row_indices_);
  }
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }

  if (read_schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(read_schema_));
    scan_builder.WithOutputSchema(c_arrow_schema);
  }

  if (parsed_predicate_) {
    scan_builder.WithFilter(*parsed_predicate_);
  }

  ARROW_RETURN_NOT_OK(apply_read_plan_selection(&scan_builder,
                                                VortexReadPlan{
                                                    .op =
                                                        VortexReadPlan::RangeScan{
                                                            .ranges = {RowRange{.start = row_start, .end = row_end}},
                                                        },
                                                    .apply_predicate = false,
                                                },
                                                vxfile_->RowCount(), path_));
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
  if (row_indices.empty()) {
    return arrow::Status::Invalid(fmt::format("out of row range[0, {}].", vxfile_->RowCount()));
  }

  // Take already carries its row selection by offsets. Predicates are only
  // applied by explicit Vortex read plans to keep take semantics positional.
  ARROW_ASSIGN_OR_RAISE(auto array_stream, read_with_plan(VortexReadPlan{
                                               .op = VortexReadPlan::Take{.row_indices = row_indices},
                                               .apply_predicate = false,
                                           }));
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

uint64_t VortexFormatReader::total_mem_usage() {
  auto sizes = vxfile_->GetUncompressedSizes();
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

}  // namespace milvus_storage::vortex
