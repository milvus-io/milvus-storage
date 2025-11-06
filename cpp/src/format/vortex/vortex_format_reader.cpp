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
#ifdef BUILD_VORTEX_BRIDGE
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "bridgeimpl.hpp"

#include <format>
#include <string>
#include <iostream>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/status.h>
#include <arrow/result.h>

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

static inline expr::Expr build_projection(const std::vector<std::string>& ncs) {
  return expr::select(std::vector<std::string_view>(ncs.begin(), ncs.end()), expr::root());
}

VortexFormatReader::VortexFormatReader(const ObjectStoreWrapper& obsw_ref,
                                       const std::shared_ptr<arrow::Schema>& schema,
                                       const std::string& path,
                                       std::vector<std::string> needed_columns)
    : obsw_ref_(obsw_ref),
      // if ::Open throws exception, current memory still clear
      vxfile_(std::move(VortexFile::Open(obsw_ref_, path))),
      proj_cols_(std::move(needed_columns)),
      schema_(schema) {
  assert(schema_);
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

arrow::Result<std::shared_ptr<arrow::ChunkedArray>> VortexFormatReader::read(uint64_t row_start, uint64_t row_end) {
  ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(schema_));
  auto array_stream = vxfile_.CreateScanBuilder()
                          .WithProjection(build_projection(proj_cols_))
                          .WithRowRange(row_start, row_end)
                          .WithOutputSchema(c_arrow_schema)
                          .IntoStream();

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  return chunkedarray;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::take(const std::vector<int64_t>& row_indices) {
  ARROW_ASSIGN_OR_RAISE(auto c_arrow_schema, export_c_arrow_schema(schema_));
  auto array_stream = vxfile_.CreateScanBuilder()
                          .WithProjection(build_projection(proj_cols_))
                          .WithIncludeByIndex((const uint64_t*)row_indices.data(), row_indices.size())
                          .WithOutputSchema(c_arrow_schema)
                          .IntoStream();

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  // out of range
  if (chunkedarray->num_chunks() == 0) {
    return arrow::Status::Invalid("out of row range[0, ", std::to_string(vxfile_.RowCount()), "].");
  }
  assert(chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

uint64_t VortexFormatReader::total_mem_usage() {
  auto sizes = vxfile_.GetUncompressedSizes();
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

uint64_t VortexFormatReader::mem_usage(size_t idx_in_column_group) {
  auto sizes = vxfile_.GetUncompressedSizes();
  if (idx_in_column_group >= static_cast<int64_t>(sizes.size())) {
    return 0;
  }
  return sizes[idx_in_column_group] == UINT64_MAX ? 0 : sizes[idx_in_column_group];
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE