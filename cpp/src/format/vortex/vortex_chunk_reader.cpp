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
#include "milvus-storage/format/vortex/vortex_chunk_reader.h"

#include <format>
#include <string>
#include <iostream>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/table.h>          // for arrow::Table
#include <arrow/status.h>

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

VortexChunkReader::VortexChunkReader(std::shared_ptr<ObjectStoreWrapper> fs,
                                     std::shared_ptr<arrow::Schema> schema,
                                     const std::vector<std::string>& paths,
                                     const std::vector<std::string>& needed_columns,
                                     const api::Properties& properties)
    : obsw_(std::move(fs)),
      number_of_chunks_(paths.size()),
      schema_(std::move(schema)),
      proj_cols_(std::move(needed_columns)),
      properties_(properties),
      paths_(std::move(paths)) {
  assert(paths_.empty() == false);
}

VortexChunkReader::~VortexChunkReader() {
  // make sure raw reference is released before obsw_ is destructed
  for (auto& vxfile : vxfiles_) {
    vxfile.reset();
  }
  vxfiles_.clear();
  obsw_.reset();
}

arrow::Status VortexChunkReader::open() {
  size_t last_offset = 0;
  vxfiles_.reserve(number_of_chunks_);
  idx_offsets_.reserve(number_of_chunks_ + 1);
  idx_offsets_.emplace_back(last_offset);

  for (const auto& path : paths_) {
    auto format_reader = std::make_unique<VortexFormatReader>(*(obsw_.get()),  // unsafe, but ok here
                                                              path, proj_cols_);
    last_offset += format_reader->rows();
    idx_offsets_.emplace_back(last_offset);
    vxfiles_.emplace_back(std::move(format_reader));
  }

  assert(idx_offsets_.size() == vxfiles_.size() + 1 && number_of_chunks_ == vxfiles_.size());

  // already opened in constructor
  return arrow::Status::OK();
}

static inline expr::Expr build_projection(const std::vector<std::string>& ncs) {
  return expr::select(std::vector<std::string_view>(ncs.begin(), ncs.end()), expr::root());
}

arrow::Result<std::vector<int64_t>> VortexChunkReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;

  for (const auto& row_index : row_indices) {
    auto it = std::lower_bound(idx_offsets_.begin(), idx_offsets_.end(), (size_t)row_index);
    if (it == idx_offsets_.end()) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }
    auto chunk_index = -1;
    if (*it == row_index) {
      chunk_index = std::distance(idx_offsets_.begin(), it);
    } else if (it != idx_offsets_.begin()) {
      chunk_index = std::distance(idx_offsets_.begin(), it - 1);
    }

    assert(chunk_index >= 0 && chunk_index < number_of_chunks_);

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::get_chunk(int64_t chunk_index) {
  if (chunk_index < 0 || chunk_index >= number_of_chunks_) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(number_of_chunks_));
  }

  return vxfiles_[chunk_index]->readall();
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> VortexChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs(chunk_indices.size());
  for (int i = 0; i < chunk_indices.size(); i++) {
    ARROW_ASSIGN_OR_RAISE(auto rb, get_chunk(chunk_indices[i]));
    rbs.emplace_back(rb);
  }

  return rbs;
}

arrow::Result<std::vector<std::vector<int64_t>>> VortexChunkReader::calc_ridxs_in_chunks(
    const std::vector<int64_t>& row_indices) {
  std::vector<std::vector<int64_t>> result(number_of_chunks_);
  for (const auto& row_index : row_indices) {
    auto it = std::lower_bound(idx_offsets_.begin(), idx_offsets_.end(), (size_t)row_index);
    if (it == idx_offsets_.end()) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }

    auto chunk_index = -1;
    if (*it == row_index) {
      chunk_index = std::distance(idx_offsets_.begin(), it);
    } else if (it != idx_offsets_.begin()) {
      chunk_index = std::distance(idx_offsets_.begin(), it - 1);
    }

    assert(chunk_index >= 0 && chunk_index < number_of_chunks_);

    int64_t local_ridx = row_index - idx_offsets_[chunk_index];
    result[chunk_index].emplace_back(local_ridx);
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::take(const std::vector<int64_t>& row_indices) {
  std::vector<std::vector<int64_t>> offset_in_chunks;
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  std::shared_ptr<arrow::RecordBatch> record_batch = nullptr;

  // calc row idxs in each chunk
  ARROW_ASSIGN_OR_RAISE(offset_in_chunks, calc_ridxs_in_chunks(row_indices));
  assert(offset_in_chunks.size() == vxfiles_.size());

  for (size_t chunk_idx = 0; chunk_idx < offset_in_chunks.size(); chunk_idx++) {
    if (offset_in_chunks[chunk_idx].empty()) {
      continue;
    }

    // no need unique the row idxs
    std::sort(offset_in_chunks[chunk_idx].begin(), offset_in_chunks[chunk_idx].end());

    ARROW_ASSIGN_OR_RAISE(record_batch, vxfiles_[chunk_idx]->take(offset_in_chunks[chunk_idx]))
    rbs.emplace_back(record_batch);
  }

  // collapse row slices
  ARROW_ASSIGN_OR_RAISE(auto combined_table, arrow::Table::FromRecordBatches(rbs));
  ARROW_ASSIGN_OR_RAISE(auto combined_batch, combined_table->CombineChunksToBatch());
  return combined_batch;
}

arrow::Result<int64_t> VortexChunkReader::get_chunk_size(int64_t chunk_index) {
  if (chunk_index < 0 || chunk_index >= (number_of_chunks_)) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(number_of_chunks_));
  }
  // no implements
  // FIXME: return actual size
  return 0;
}

arrow::Result<int64_t> VortexChunkReader::get_chunk_rows(int64_t chunk_index) {
  if (chunk_index < 0 || chunk_index >= (number_of_chunks_)) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(number_of_chunks_));
  }
  return idx_offsets_[chunk_index + 1] - idx_offsets_[chunk_index];
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE