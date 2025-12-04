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

#include "milvus-storage/format/column_group_reader.h"

#include <unordered_map>
#include <vector>
#include <algorithm>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>

#include "milvus-storage/format/parquet/parquet_chunk_reader.h"
#include "milvus-storage/format/vortex/vortex_chunk_reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace internal::api {

// TODO: move to the file_system.h
using namespace milvus_storage::parquet;
static inline arrow::Result<milvus_storage::ArrowFileSystemPtr> create_arrow_file_system(
    const milvus_storage::ArrowFileSystemConfig& fs_config) {
  auto& fs_cache = milvus_storage::LRUCache<milvus_storage::ArrowFileSystemConfig,
                                            milvus_storage::ArrowFileSystemPtr>::getInstance();
  return fs_cache.get(fs_config, milvus_storage::CreateArrowFileSystem);
}

#ifdef BUILD_VORTEX_BRIDGE
using namespace milvus_storage::vortex;
#endif  // BUILD_VORTEX_BRIDGE

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

std::string RowGroupInfo::ToString() const {
  std::stringstream ss;
  ss << "RowGroupInfo{"
     << "start_offset=" << start_offset << ", end_offset=" << end_offset << ", memory_size=" << memory_size << "}";
  return ss.str();
}

ColumnGroupReaderImpl::ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                                             const milvus_storage::api::Properties& properties,
                                             const std::vector<std::string>& needed_columns,
                                             const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      num_of_files_(column_group->files.size()) {}

arrow::Status ColumnGroupReaderImpl::open() {
  // create internal reader
  ARROW_ASSIGN_OR_RAISE(internal_, ColumnGroupReaderInternal::create(schema_, column_group_, properties_,
                                                                     needed_columns_, key_retriever_));
  // open internal reader
  ARROW_ASSIGN_OR_RAISE(std::tie(chunk_infos_, row_group_infos_), internal_->open());

  total_rows_ = 0;
  for (const auto& chunk_info : chunk_infos_) {
    total_rows_ += chunk_info.number_of_rows;
  }

  return arrow::Status::OK();
}

size_t ColumnGroupReaderImpl::total_number_of_chunks() const {
  assert(internal_);
  return chunk_infos_.size();
}

size_t ColumnGroupReaderImpl::total_rows() const {
  assert(internal_);
  return total_rows_;
}

arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  assert(internal_);
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(chunk_infos_.begin(), chunk_infos_.end(), row_index,
                               [](int64_t a, const ChunkInfo& b) { return a < b.global_row_end; });
    auto chunk_index = std::distance(chunk_infos_.begin(), it);
    if (chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl::get_chunk(int64_t chunk_index) {
  assert(internal_);
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  ARROW_ASSIGN_OR_RAISE(auto table, internal_->get_chunk(chunk_info.file_index, row_group_infos_[chunk_info.file_index],
                                                         chunk_info.row_group_index_in_file));
  assert(table);

  if (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != table->num_rows()) {
    table = table->Slice(chunk_info.row_offset_in_row_group, chunk_info.number_of_rows);
  }

  return milvus_storage::ConvertTableToRecordBatch(table, false);
}

struct ChunkMapEntry {
  std::vector<int> row_group_indices;
  std::shared_ptr<arrow::Table> table;
  std::vector<int64_t> row_group_offsets;
};

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
  assert(internal_);
  std::vector<std::vector<int>> row_groups_in_files(num_of_files_);
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  std::vector<std::optional<ChunkMapEntry>> chunk_map(num_of_files_);

  // 1. Grouping row groups by file
  //
  // the row_groups_in_files will stored as
  // row_groups_in_files[file_index] = [row_group_index_in_file]
  // example:
  //   file 0: row group 0, 1, 1, 2
  //   file 1: row group 0, 1
  //   file 2: row group 0
  //   file 0: row group 1, 2 <- same file 0, but we have to read it again
  //   then row_groups_in_files = [[0, 1, 1, 2], [0, 1], [0], [1, 2]]
  for (int64_t chunk_index : chunk_indices) {
    if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                    std::to_string(chunk_infos_.size()));
    }

    auto row_group = chunk_infos_[chunk_index];
    row_groups_in_files[row_group.file_index].emplace_back(row_group.row_group_index_in_file);
  }

  // 2. Deduplicate row groups to avoid reading the same row group multiple times
  //    Also sort the row groups to make it continuous
  //
  // example:
  //   row_groups_in_files = [[0, 1, 1, 2], [0, 1], [0], [1, 2]]
  //   then row_groups_in_files = [[0, 1, 2], [0, 1], [0], [1, 2]]
  for (auto& row_groups : row_groups_in_files) {
    std::sort(row_groups.begin(), row_groups.end());
    row_groups.erase(std::unique(row_groups.begin(), row_groups.end()), row_groups.end());
  }

  // 3. Read row groups and store them in chunk_map
  //
  // example:
  //   row_groups_in_files = [[0, 1, 2], [0, 1], [0], [1, 2]]
  //   then chunk_map = [{row_group_indices=[0, 1, 2], table=table0},
  //                     {row_group_indices=[0, 1], table=table1},
  //                     {row_group_indices=[0], table=table2},
  //                     {row_group_indices=[1, 2], table=table3}]
  for (size_t file_idx = 0; file_idx < row_groups_in_files.size(); ++file_idx) {
    const auto& row_group_indices = row_groups_in_files[file_idx];
    if (row_group_indices.empty()) {
      continue;
    }

    ARROW_ASSIGN_OR_RAISE(auto table, internal_->get_chunks(file_idx, row_group_infos_[file_idx], row_group_indices));

    // calculate the offset of each row group
    std::vector<int64_t> offsets;
    int64_t current_offset = 0;
    const auto& row_group_infos = row_group_infos_[file_idx];
    for (int rg_idx : row_group_indices) {
      offsets.emplace_back(current_offset);
      current_offset += row_group_infos[rg_idx].end_offset - row_group_infos[rg_idx].start_offset;
    }

    chunk_map[file_idx] =
        ChunkMapEntry{.row_group_indices = row_group_indices, .table = table, .row_group_offsets = offsets};
  }

  // 4. Remapping chunk indices with row groups also slice the record batch with the chunk info
  // notice that: the range of row groups may not match the range from chunk info
  // example:
  //   chunk_indices = [3, 1, 2, 0]
  //   then chunk_map = [{row_group_indices=[1, 2], table=table3, row_group_offsets=[0, 100, 200]},
  //                     {row_group_indices=[0, 1], table=table1, row_group_offsets=[0, 100]},
  //                     {row_group_indices=[0], table=table2, row_group_offsets=[0]},
  //                     {row_group_indices=[0, 1, 2], table=table0, row_group_offsets=[100, 200]}
  //   then result = [table3, table1, table2, table0]
  for (int64_t chunk_index : chunk_indices) {
    auto row_group = chunk_infos_[chunk_index];

    // current file have not been requested
    if (!chunk_map[row_group.file_index].has_value()) {
      continue;
    }

    const auto& entry = chunk_map[row_group.file_index].value();

    // Find the index of the row group in the read row groups
    auto it = std::lower_bound(entry.row_group_indices.begin(), entry.row_group_indices.end(),
                               row_group.row_group_index_in_file);
    if (it == entry.row_group_indices.end() || *it != row_group.row_group_index_in_file) {
      return arrow::Status::Invalid("Row group index not found in chunk map entry");
    }
    size_t rg_idx_in_read_list = std::distance(entry.row_group_indices.begin(), it);

    // Calculate the offset in the table
    int64_t offset_in_table = entry.row_group_offsets[rg_idx_in_read_list];
    offset_in_table += row_group.row_offset_in_row_group;

    auto sliced_table = entry.table->Slice(offset_in_table, row_group.number_of_rows);

    ARROW_ASSIGN_OR_RAISE(auto batch, milvus_storage::ConvertTableToRecordBatch(sliced_table, false));
    result.emplace_back(batch);
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl::take(
    const std::vector<int64_t>& row_indices) {
  assert(internal_);
  return arrow::Status::NotImplemented("take is not implemented");
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_size(int64_t chunk_index) {
  assert(internal_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].avg_memory_size;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_rows(int64_t chunk_index) {
  assert(internal_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].number_of_rows;
}

arrow::Result<std::unique_ptr<ColumnGroupReaderInternal>> ColumnGroupReaderInternal::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  milvus_storage::ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
  ARROW_ASSIGN_OR_RAISE(auto file_system, create_arrow_file_system(fs_config));
  if (column_group->format == LOON_FORMAT_PARQUET) {
    return std::make_unique<ParquetChunkReader>(file_system, column_group, properties, needed_columns, key_retriever);
  }
#ifdef BUILD_VORTEX_BRIDGE
  else if (column_group->format == LOON_FORMAT_VORTEX) {
    return std::make_unique<VortexChunkReader>(file_system, schema, column_group, properties, needed_columns);
  }
#endif  // BUILD_VORTEX_BRIDGE
  else {
    return arrow::Status::Invalid("Unsupported file format: " + column_group->format);
  }
}

}  // namespace internal::api