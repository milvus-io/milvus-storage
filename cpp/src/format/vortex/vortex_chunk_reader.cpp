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

VortexChunkReader::VortexChunkReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                     const std::shared_ptr<arrow::Schema>& schema,
                                     const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                                     const api::Properties& properties,
                                     const std::vector<std::string>& needed_columns)
    : fs_holder_(std::make_shared<FileSystemWrapper>(fs)),
      schema_(schema),
      proj_cols_(std::move(needed_columns)),
      properties_(properties),
      column_group_(column_group),
      cg_files_(column_group->files),
      vortex_readers_(),
      logical_chunk_rows_(0) {
  assert(!cg_files_.empty());
}

VortexChunkReader::~VortexChunkReader() {
  // make sure raw reference is released before fs_holder_ is destructed
  vortex_readers_.clear();
  fs_holder_.reset();
}

static std::vector<RowGroupInfo> get_row_group_infos(uint64_t memory_usage_in_file,
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

arrow::Result<std::pair<std::vector<ChunkInfo>, std::vector<std::vector<RowGroupInfo>>>> VortexChunkReader::open() {
  // should not be called twice
  assert(vortex_readers_.empty());

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_VORTEX_CHUNK_ROWS));
  std::vector<ChunkInfo> chunk_infos;
  std::vector<std::vector<RowGroupInfo>> all_row_group_infos;

  uint64_t last_offset = 0;  // the offset of all files
  // load vortex files and build chunk infos
  for (size_t file_idx = 0; file_idx < cg_files_.size(); file_idx++) {
    const auto& cg_file = cg_files_[file_idx];
    const auto& file_path = cg_file.path;
    std::shared_ptr<VortexFormatReader> file_reader =
        std::make_shared<VortexFormatReader>(fs_holder_, schema_,  // unsafe, but ok here
                                             file_path, proj_cols_);
    vortex_readers_.emplace_back(file_reader);

    // get the row range infos, maybe we need do the pre-cache?
    auto mem_usage_in_file = file_reader->total_mem_usage();
    auto rows_in_file = file_reader->rows();
    std::vector<RowGroupInfo> row_group_infos = get_row_group_infos(
        mem_usage_in_file, rows_in_file, recalc_row_ranges(file_reader->row_ranges(), logical_chunk_rows_));

    uint64_t offset_in_file = 0;  // the offset of current file
    if (cg_file.start_index.has_value() && cg_file.end_index.has_value()) {
      int64_t start_index = cg_file.start_index.value();
      int64_t end_index = cg_file.end_index.value();

      if (UNLIKELY(start_index < 0 || end_index < 0 || start_index >= end_index)) {
        return arrow::Status::Invalid("Invalid start/end index", "[path=", file_path, "] [start_index=", start_index,
                                      "] [end_index=", end_index, "]");
      }

      for (size_t j = 0; j < row_group_infos.size(); ++j) {
        uint64_t rg_start = row_group_infos[j].start_offset;
        uint64_t rg_end = row_group_infos[j].end_offset;

        // calculate the overlap range
        uint64_t overlap_start = std::max((uint64_t)start_index, rg_start);
        uint64_t overlap_end = std::min((uint64_t)end_index, rg_end);

        // if the overlap range is valid, create the chunk info
        if (overlap_start < overlap_end) {
          offset_in_file += overlap_end - overlap_start;
          chunk_infos.emplace_back(std::move(ChunkInfo{
              .file_index = file_idx,
              .row_offset_in_row_group = overlap_start - rg_start,
              .row_offset_in_file = overlap_start,
              .number_of_rows = overlap_end - overlap_start,
              .row_group_index_in_file = j,

              .global_row_end = last_offset + offset_in_file,
              // if memory_usage_in_file is 0 will be fine
              .avg_memory_size =
                  rows_in_file == 0 ? 0 : mem_usage_in_file * (overlap_end - overlap_start) / rows_in_file,
          }));
        }
      }

    } else {
      for (size_t j = 0; j < row_group_infos.size(); ++j) {
        offset_in_file += row_group_infos[j].end_offset - row_group_infos[j].start_offset;
        chunk_infos.emplace_back(std::move(ChunkInfo{
            .file_index = file_idx,
            .row_offset_in_row_group = 0,
            .row_offset_in_file = row_group_infos[j].start_offset,
            .number_of_rows = (row_group_infos[j].end_offset - row_group_infos[j].start_offset),
            .row_group_index_in_file = j,

            .global_row_end = last_offset + offset_in_file,
            // if memory_usage_in_file is 0 will be fine
            .avg_memory_size = rows_in_file == 0
                                   ? 0
                                   : mem_usage_in_file *
                                         (row_group_infos[j].end_offset - row_group_infos[j].start_offset) /
                                         rows_in_file,
        }));
      }
    }

    last_offset += rows_in_file;
    all_row_group_infos.emplace_back(std::move(row_group_infos));
  }

  return std::make_pair(chunk_infos, all_row_group_infos);
}

// TODO: vortex no need read full row group if start/end index is specified?
arrow::Result<std::shared_ptr<arrow::Table>> VortexChunkReader::get_chunk(
    size_t file_index, const std::vector<RowGroupInfo>& row_group_info, const int& rg_index_in_file) {
  assert(!vortex_readers_.empty());

  ARROW_ASSIGN_OR_RAISE(auto chunkedarray,
                        vortex_readers_[file_index]->read(row_group_info[rg_index_in_file].start_offset,
                                                          row_group_info[rg_index_in_file].end_offset));

  // won't split as multi-chunk in vortex
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);

  ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0)));
  return arrow::Table::FromRecordBatches({rb});
}

arrow::Result<std::shared_ptr<arrow::Table>> VortexChunkReader::get_chunks(
    size_t file_index, const std::vector<RowGroupInfo>& row_group_info, const std::vector<int>& rg_indices_in_file) {
  assert(!vortex_readers_.empty());
  assert(file_index < vortex_readers_.size());
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
    const auto& start_rg_info = row_group_info[rg_range.first];
    const auto& end_rg_info = row_group_info[rg_range.second];

    ARROW_ASSIGN_OR_RAISE(auto chunked_array,
                          vortex_readers_[file_index]->read(start_rg_info.start_offset, end_rg_info.end_offset));
    // assign to rbs
    for (size_t j = 0; j < chunked_array->num_chunks(); ++j) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunked_array->chunk(j)));
      rbs.emplace_back(rb);
    }
  }

  return arrow::Table::FromRecordBatches(rbs);
}

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE