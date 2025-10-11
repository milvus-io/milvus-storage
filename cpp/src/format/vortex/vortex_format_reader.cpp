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

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

VortexFormatReader::VortexFormatReader(const ObjectStoreWrapper& obsw_ref,
                                       const std::string& path,
                                       std::vector<std::string> needed_columns)
    : obsw_ref_(obsw_ref),
      vxfile_(std::move(VortexFile::Open(obsw_ref_, path))),
      proj_cols_(std::move(needed_columns)) {}

static inline expr::Expr build_projection(const std::vector<std::string>& ncs) {
  return expr::select(std::vector<std::string_view>(ncs.begin(), ncs.end()), expr::root());
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::readall() {
  auto array_stream = vxfile_.CreateScanBuilder().WithProjection(build_projection(proj_cols_)).IntoStream();

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  assert(chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexFormatReader::take(const std::vector<int64_t>& row_indices) {
  auto array_stream = vxfile_.CreateScanBuilder()
                          .WithProjection(build_projection(proj_cols_))
                          .WithIncludeByIndex((const uint64_t*)row_indices.data(), row_indices.size())
                          .IntoStream();

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  // out of range
  if (chunkedarray->num_chunks() == 0) {
    return arrow::Status::Invalid("out of row range[0, ", std::to_string(vxfile_.RowCount()), "].");
  }
  assert(chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE