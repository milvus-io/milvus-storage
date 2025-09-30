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
#include "milvus-storage/format/vortex/vortex_reader.h"
#include "reader_ffi.hpp"

#include <format>
#include <string>
#include <iostream>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include "milvus-storage/common/status.h"

namespace milvus_storage::vortex {

VortexChunkReader::VortexChunkReader(ArrowFileSystemConfig config,
                                     std::shared_ptr<arrow::Schema> schema,
                                     const std::string& path,
                                     std::vector<std::string> needed_columns)
    : obsw_(std::move(ObjectStoreWrapper2::OpenObjectStore(std::string("unused"),
                                                           config.address,
                                                           config.access_key_id,
                                                           config.access_key_value,
                                                           config.region,
                                                           config.bucket_name))),
      vxfile_(std::move(VortexFile::Open(obsw_, path))),
      schema_(std::move(schema)),
      proj_cols_(std::move(needed_columns)) {}

static inline expr::Expr build_projection(const std::vector<std::string>& ncs) {
  return expr::select(std::vector<std::string_view>(ncs.begin(), ncs.end()), expr::root());
}

arrow::Result<std::vector<int64_t>> VortexChunkReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
#ifndef NDEBUG
  for (const auto idx : row_indices) {
    assert(idx == 0);
  }
#endif

  return std::vector<int64_t>(row_indices.size(), 0);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::get_chunk([[maybe_unused]] int64_t chunk_index) {
  assert(chunk_index == 0);

  auto array_stream = vxfile_.CreateScanBuilder().WithProjection(build_projection(proj_cols_)).IntoStream();

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  assert(chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> VortexChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
#ifndef NDEBUG
  for (const auto idx : chunk_indices) {
    assert(idx == 0);
  }
#endif
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs(chunk_indices.size());
  for (int i = 0; i < chunk_indices.size(); i++) {
    ARROW_ASSIGN_OR_RAISE(auto rb, get_chunk(chunk_indices[i]));
    rbs.emplace_back(rb);
  }

  return rbs;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::take(const std::vector<int64_t>& row_indices) {
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

arrow::Result<int64_t> VortexChunkReader::get_chunk_size(int64_t chunk_index) {
  assert(chunk_index == 0);
  // not implements
  return 0;
}

arrow::Result<int64_t> VortexChunkReader::get_chunk_rows(int64_t chunk_index) {
  assert(chunk_index == 0);
  return vxfile_.RowCount();
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE