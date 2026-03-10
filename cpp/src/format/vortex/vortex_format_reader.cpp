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
#include <memory>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <folly/futures/Promise.h>

namespace milvus_storage::vortex {

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

static inline arrow::Result<ArrowSchema> export_c_arrow_schema(std::shared_ptr<arrow::Schema> schema) {
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

static std::vector<RowGroupInfo> create_row_group_infos(uint64_t memory_usage_in_file,
                                                        uint64_t rows_in_file,
                                                        const std::vector<uint64_t>& row_ranges) {
  if (rows_in_file == 0) {
    return std::vector<RowGroupInfo>();
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

VortexFormatReader::VortexFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                       const std::shared_ptr<arrow::Schema>& schema,
                                       const std::string& path,
                                       const milvus_storage::api::Properties& properties,
                                       const std::vector<std::string>& needed_columns)
    : fs_holder_(std::make_shared<FileSystemWrapper>(fs)),
      proj_cols_(std::move(needed_columns)),
      path_(path),
      schema_(schema),
      properties_(properties),
      vxfile_(nullptr) {}

arrow::Status VortexFormatReader::open() {
  assert(!vxfile_);

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  if (schema_ && schema_->num_fields() == 0) {
    schema_ = nullptr;
  }
  vxfile_ = VortexFile::OpenUnique((uint8_t*)fs_holder_.get(), path_);

  row_group_infos_ =
      create_row_group_infos(total_mem_usage(), rows(), recalc_row_ranges(row_ranges(), logical_chunk_rows_));

  return arrow::Status::OK();
}

arrow::Result<std::vector<RowGroupInfo>> VortexFormatReader::get_row_group_infos() {
  assert(vxfile_);
  return row_group_infos_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::get_chunk(const int& row_group_index) {
  assert(vxfile_);
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, blocking_read(row_group_infos_[row_group_index].start_offset,
                                                         row_group_infos_[row_group_index].end_offset));
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);

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

  if (schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(schema_));
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

  if (schema_) {
    ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(schema_));
    scan_builder.WithOutputSchema(c_arrow_schema);
  }

  scan_builder.WithIncludeByIndex((const uint64_t*)row_indices.data(), row_indices.size());

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

// ---------------------------------------------------------------------------
// Truly-async read path – runs on the Rust Tokio runtime, resolves a
// folly::Promise via extern "C" callback.
// ---------------------------------------------------------------------------

/// Shared context allocated on the heap and owned by the callback.
template <typename T>
struct VortexAsyncContext {
  folly::Promise<arrow::Result<T>> promise;
  ArrowArrayStream stream;  // storage written to by Rust
};

/// Callback for take_async(), invoked exactly once from a Tokio worker thread
/// when the Rust-side async scan completes.
///
/// On success (error_msg == nullptr):
///   1. Import the FFI ArrowArrayStream (written by Rust) as a ChunkedArray.
///   2. Convert each chunk (StructArray) into a RecordBatch.
///   3. Assemble the RecordBatches into a Table and resolve the Promise.
///
/// On failure (error_msg != nullptr):
///   Set the Promise to an IOError and free the Rust-allocated error string.
///
/// Ownership: this function takes ownership of ctx_raw via unique_ptr;
/// the VortexAsyncContext is automatically freed when the callback returns.
static void vortex_take_async_callback(void* ctx_raw,
                                       ArrowArrayStream* /*out_stream*/,
                                       const char* error_msg) {
  // Reclaim ownership of the context (allocated in take_async via make_unique + release).
  std::unique_ptr<VortexAsyncContext<std::shared_ptr<arrow::Table>>> ctx(
      static_cast<VortexAsyncContext<std::shared_ptr<arrow::Table>>*>(ctx_raw));

  if (error_msg) {
    // Rust reported an error — propagate as IOError and free the C string.
    ctx->promise.setValue(arrow::Status::IOError(std::string(error_msg)));
    vortex_free_error_string(const_cast<char*>(error_msg));
    return;
  }

  // Import the FFI stream that Rust wrote into ctx->stream.
  auto result = arrow::ImportChunkedArray(&ctx->stream);
  if (!result.ok()) {
    ctx->promise.setValue(result.status());
    return;
  }

  auto chunkedarray = result.ValueUnsafe();
  if (chunkedarray->num_chunks() == 0) {
    ctx->promise.setValue(arrow::Status::Invalid("take_async: empty result"));
    return;
  }

  // Convert each StructArray chunk into a RecordBatch.
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  rbs.reserve(chunkedarray->num_chunks());
  for (int i = 0; i < chunkedarray->num_chunks(); ++i) {
    auto rb = arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i));
    if (!rb.ok()) {
      ctx->promise.setValue(rb.status());
      return;
    }
    rbs.emplace_back(rb.ValueUnsafe());
  }

  // Assemble into a Table and resolve the SemiFuture.
  ctx->promise.setValue(arrow::Table::FromRecordBatches(rbs));
}

/// Callback for read_with_range_async(), invoked exactly once from a Tokio
/// worker thread when the Rust-side async scan completes.
///
/// On success: import the FFI ArrowArrayStream as a RecordBatchReader and
///             resolve the Promise.
/// On failure: set the Promise to an IOError and free the Rust-allocated
///             error string.
///
/// Ownership: same as vortex_take_async_callback — ctx_raw is reclaimed
/// via unique_ptr and freed automatically on return.
static void vortex_read_range_async_callback(void* ctx_raw,
                                             ArrowArrayStream* /*out_stream*/,
                                             const char* error_msg) {
  // Reclaim ownership of the context (allocated in read_with_range_async via make_unique + release).
  std::unique_ptr<VortexAsyncContext<std::shared_ptr<arrow::RecordBatchReader>>> ctx(
      static_cast<VortexAsyncContext<std::shared_ptr<arrow::RecordBatchReader>>*>(ctx_raw));

  if (error_msg) {
    // Rust reported an error — propagate as IOError and free the C string.
    ctx->promise.setValue(arrow::Status::IOError(std::string(error_msg)));
    vortex_free_error_string(const_cast<char*>(error_msg));
    return;
  }

  // Import the FFI stream as a RecordBatchReader and resolve the SemiFuture.
  ctx->promise.setValue(arrow::ImportRecordBatchReader(&ctx->stream));
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>
VortexFormatReader::take_async(const std::vector<int64_t>& row_indices) {
  assert(vxfile_);

  auto scan_builder = vxfile_->CreateScanBuilder();
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }
  if (schema_) {
    auto c_schema_result = export_c_arrow_schema(schema_);
    if (!c_schema_result.ok()) {
      return folly::makeSemiFuture(
          arrow::Result<std::shared_ptr<arrow::Table>>(c_schema_result.status()));
    }
    auto c_schema = c_schema_result.MoveValueUnsafe();
    scan_builder.WithOutputSchema(c_schema);
  }

  scan_builder.WithIncludeByIndex(
      reinterpret_cast<const uint64_t*>(row_indices.data()), row_indices.size());

  uintptr_t handle = std::move(scan_builder).IntoRawHandle();

  auto ctx = std::make_unique<VortexAsyncContext<std::shared_ptr<arrow::Table>>>();
  auto sf = ctx->promise.getSemiFuture();
  auto* raw_ctx = ctx.release();  // ownership transferred to callback

  vortex_scan_collect_async(
      handle, &raw_ctx->stream, vortex_take_async_callback, static_cast<void*>(raw_ctx));

  return sf;
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>>
VortexFormatReader::read_with_range_async(uint64_t start_offset, uint64_t end_offset) {
  assert(vxfile_);

  auto scan_builder = vxfile_->CreateScanBuilder();
  if (!proj_cols_.empty()) {
    scan_builder.WithProjection(build_projection(proj_cols_));
  }
  if (schema_) {
    auto c_schema_result = export_c_arrow_schema(schema_);
    if (!c_schema_result.ok()) {
      return folly::makeSemiFuture(
          arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>(c_schema_result.status()));
    }
    auto c_schema = c_schema_result.MoveValueUnsafe();
    scan_builder.WithOutputSchema(c_schema);
  }

  scan_builder.WithRowRange(start_offset, end_offset);

  uintptr_t handle = std::move(scan_builder).IntoRawHandle();

  auto ctx = std::make_unique<VortexAsyncContext<std::shared_ptr<arrow::RecordBatchReader>>>();
  auto sf = ctx->promise.getSemiFuture();
  auto* raw_ctx = ctx.release();  // ownership transferred to callback

  vortex_scan_collect_async(
      handle, &raw_ctx->stream, vortex_read_range_async_callback, static_cast<void*>(raw_ctx));

  return sf;
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