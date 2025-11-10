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

#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

VortexChunkReader::VortexChunkReader(std::shared_ptr<ObjectStoreWrapper> fs,
                                     const std::shared_ptr<arrow::Schema>& schema,
                                     const std::vector<std::string>& paths,
                                     const api::Properties& properties,
                                     const std::vector<std::string>& needed_columns)
    : obsw_(std::move(fs)),
      schema_(schema),
      proj_cols_(std::move(needed_columns)),
      properties_(properties),
      paths_(std::move(paths)),
      vxfiles_(paths.size()),
      logical_chunk_rows_(0),
      rginfos_(),
      offsets_in_paths_(paths.size() + 1),
      total_rows_(0) {
  assert(paths_.empty() == false);
}

size_t VortexChunkReader::total_number_of_chunks() const {
  assert(!rginfos_.empty());
  return rginfos_.size();
}

size_t VortexChunkReader::total_rows() const {
  assert(!rginfos_.empty());
  return total_rows_;
}

VortexChunkReader::~VortexChunkReader() {
  // make sure raw reference is released before obsw_ is destructed
  for (auto& vxfile : vxfiles_) {
    vxfile.reset();
  }
  vxfiles_.clear();
  obsw_.reset();
}

static arrow::Result<std::vector<std::vector<int64_t>>> calc_idxs_in_paths(
    const std::vector<int64_t>& row_indices, const std::vector<uint64_t>& offsets_in_paths) {
  std::vector<std::vector<int64_t>> result(offsets_in_paths.size());
  for (const auto& row_index : row_indices) {
    auto it = std::lower_bound(offsets_in_paths.begin(), offsets_in_paths.end(), (size_t)row_index);
    if (it == offsets_in_paths.end()) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }

    auto chunk_index = -1;
    if (*it == row_index) {
      chunk_index = std::distance(offsets_in_paths.begin(), it);
    } else if (it != offsets_in_paths.begin()) {
      chunk_index = std::distance(offsets_in_paths.begin(), it - 1);
    }

    assert(chunk_index >= 0 && chunk_index < offsets_in_paths.size());

    int64_t local_ridx = row_index - offsets_in_paths[chunk_index];
    result[chunk_index].emplace_back(local_ridx);
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

arrow::Status VortexChunkReader::open() {
  // should not be called twice
  assert(rginfos_.empty());

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_VORTEX_CHUNK_ROWS));

  assert(!offsets_in_paths_.empty());
  offsets_in_paths_[0] = 0;

  size_t last_offset = 0;
  // load vortex files and build chunk infos
  for (size_t path_idx = 0; path_idx < paths_.size(); path_idx++) {
    vxfiles_[path_idx] = std::make_unique<VortexFormatReader>(*obsw_, schema_,  // unsafe, but ok here
                                                              paths_[path_idx], proj_cols_);

    auto memory_usage_in_file = vxfiles_[path_idx]->total_mem_usage();
    auto rows_in_file = vxfiles_[path_idx]->rows();
    auto row_ranges = vxfiles_[path_idx]->row_ranges();
    row_ranges = recalc_row_ranges(row_ranges, logical_chunk_rows_);

    size_t last_offset_in_file = 0;
    for (size_t rgidx = 0; rgidx < row_ranges.size(); rgidx++) {
      rginfos_.emplace_back(std::move(ChunkInfo{
          .belong_which_file = path_idx,
          .global_row_offset = last_offset_in_file + last_offset,
          .row_index_in_file = last_offset_in_file,
          .number_of_rows = row_ranges[rgidx],

          // if memory_usage_in_file is 0 will be fine
          .avg_memory_usage = memory_usage_in_file * (row_ranges[rgidx] / rows_in_file),
      }));
      last_offset_in_file += row_ranges[rgidx];
    }

    assert(last_offset_in_file == rows_in_file);
    last_offset += rows_in_file;

    assert(offsets_in_paths_.size() > path_idx + 1);
    offsets_in_paths_[path_idx + 1] = last_offset;
  }
  total_rows_ = last_offset;

#ifndef NDEBUG
  // verify total rows
  assert(!offsets_in_paths_.empty());
  assert(offsets_in_paths_.back() == total_rows_);

  // rginfos_ correctness and order
  auto verify_total_rows = 0;
  for (size_t i = 0; i < rginfos_.size(); ++i) {
    verify_total_rows += rginfos_[i].number_of_rows;
  }
  assert(verify_total_rows == total_rows_);
#endif

  // already opened in constructor
  return arrow::Status::OK();
}

arrow::Result<std::vector<int64_t>> VortexChunkReader::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;

  assert(!rginfos_.empty());
  for (int64_t row_index : row_indices) {
    // use upper_bound find the first position which global_row_offset > row_index
    auto it =
        std::upper_bound(rginfos_.begin(), rginfos_.end(), row_index,
                         [](uint64_t row_idx, const ChunkInfo& info) { return row_idx < info.global_row_offset; });
    assert(it != rginfos_.begin());

    // move to the correct chunks
    --it;

    // check row_index in range
    if (row_index >= it->global_row_offset + it->number_of_rows) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }
    auto chunk_index = std::distance(rginfos_.begin(), it);
    assert(chunk_index >= 0 && chunk_index < rginfos_.size());
    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::get_chunk(int64_t chunk_index) {
  assert(!rginfos_.empty());
  if (chunk_index < 0 || chunk_index >= rginfos_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(rginfos_.size()));
  }

  const auto& rg_info = rginfos_[chunk_index];
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray,
                        vxfiles_[rg_info.belong_which_file]->read(rg_info.row_index_in_file,
                                                                  rg_info.row_index_in_file + rg_info.number_of_rows));
  // won't split as multi-chunk in vortex
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> VortexChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  std::vector<std::vector<int>> chunks_in_files(vxfiles_.size());

  assert(!rginfos_.empty());

  for (const auto chunk_idx : chunk_indices) {
    const auto& rg_info = rginfos_[chunk_idx];
    chunks_in_files[rg_info.belong_which_file].emplace_back(chunk_idx);
  }

  for (auto& chunks_in_single_file : chunks_in_files) {
    std::vector<std::pair<uint64_t, uint64_t>> rg_idx_ranges;
    // do sort chunks_in_single_file
    std::sort(chunks_in_single_file.begin(), chunks_in_single_file.end());

    // calc continuous ranges
    // ex. [1, 2, 3, 5] -> [(1, 3), (5, 5)]
    size_t start_idx = 0;
    for (size_t i = 1; i < chunks_in_single_file.size(); ++i) {
      if (chunks_in_single_file[i] != chunks_in_single_file[i - 1] + 1) {
        rg_idx_ranges.emplace_back(chunks_in_single_file[start_idx], chunks_in_single_file[i - 1]);
        start_idx = i;
      }
    }

    if (start_idx < chunks_in_single_file.size()) {
      rg_idx_ranges.emplace_back(chunks_in_single_file[start_idx], chunks_in_single_file.back());
    }

    // begin to read continuous ranges
    // for each range, read once, then split to record batches
    // and assign to rbs
    for (const auto& rg_range : rg_idx_ranges) {
      // load continuous chunks in one read
      const auto& start_rg_info = rginfos_[rg_range.first];
      const auto& end_rg_info = rginfos_[rg_range.second];

      ARROW_ASSIGN_OR_RAISE(auto chunked_array, vxfiles_[start_rg_info.belong_which_file]->read(
                                                    start_rg_info.row_index_in_file,
                                                    end_rg_info.row_index_in_file + end_rg_info.number_of_rows));
      // assign to rbs
      for (size_t j = 0; j < chunked_array->num_chunks(); ++j) {
        ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunked_array->chunk(j)));
        rbs.emplace_back(rb);
      }
    }
  }

  return rbs;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> VortexChunkReader::take(const std::vector<int64_t>& row_indices) {
  std::vector<std::vector<int64_t>> offset_in_chunks;
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  std::shared_ptr<arrow::RecordBatch> record_batch = nullptr;

  // calc row idxs in each chunk
  ARROW_ASSIGN_OR_RAISE(offset_in_chunks, calc_idxs_in_paths(row_indices, offsets_in_paths_));
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

  // FIXME: will it do concatenation(means copy)?
  ARROW_ASSIGN_OR_RAISE(auto combined_batch, combined_table->CombineChunksToBatch());
  return combined_batch;
}

arrow::Result<uint64_t> VortexChunkReader::get_chunk_size(int64_t chunk_index) {
  assert(!rginfos_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= rginfos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(rginfos_.size()));
  }

  return rginfos_[chunk_index].avg_memory_usage;
}

arrow::Result<uint64_t> VortexChunkReader::get_chunk_rows(int64_t chunk_index) {
  assert(!rginfos_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= rginfos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: ", std::to_string(chunk_index), " out of ",
                                  std::to_string(rginfos_.size()));
  }
  return rginfos_[chunk_index].number_of_rows;
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE