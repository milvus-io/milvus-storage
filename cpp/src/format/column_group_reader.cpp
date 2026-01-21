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

#include <numeric>
#include <unordered_map>
#include <future>
#include <vector>
#include <algorithm>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include <folly/executors/IOThreadPoolExecutor.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace milvus_storage::api {

using milvus_storage::RowGroupInfo;
struct ChunkInfo {
  public:
  size_t file_index;               // current chunk belong which file
  size_t row_offset_in_row_group;  // the starting row offset of this row group in its file
  size_t row_offset_in_file;       // the starting row offset of file
  size_t number_of_rows;           // number of rows in this row group
  size_t row_group_index_in_file;  // the index of this row group in its file
  size_t global_row_end;           // the ending row offset of this row group in the whole chunk reader
  size_t avg_memory_size;          // average memory usage of this row group

  ChunkInfo() = default;
  std::string ToString() const;
};

class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever);

  ~ColumnGroupReaderImpl() = default;

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] size_t total_rows() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_size(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  size_t total_rows_;

  std::vector<std::shared_ptr<FormatReader>> format_readers_;
};  // ColumnGroupReaderImpl

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever) {
  std::unique_ptr<ColumnGroupReader> reader = nullptr;
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  // Generate the output schema with only the needed columns
  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
      auto field = schema->GetFieldByName(col_name);
      assert(field);
      fields.emplace_back(field);
    }
  }

  out_schema = std::make_shared<arrow::Schema>(fields);
  reader = std::make_unique<milvus_storage::api::ColumnGroupReaderImpl>(out_schema, column_group, properties,
                                                                        filtered_columns, key_retriever);
  ARROW_RETURN_NOT_OK(reader->open());
  return std::move(reader);
}

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

ColumnGroupReaderImpl::ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::shared_ptr<api::ColumnGroup>& column_group,
                                             const api::Properties& properties,
                                             const std::vector<std::string>& needed_columns,
                                             const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever) {}

arrow::Status ColumnGroupReaderImpl::open() {
  const auto& cg_files = column_group_->files;

  // init chunk infos
  size_t rows_in_all_files = 0;
  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto& cg_file = cg_files[file_idx];

    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return arrow::Status::Invalid("Invalid start/end index in [file_index=", file_idx, ", path=", cg_file.path, "]");
    }

    ARROW_ASSIGN_OR_RAISE(auto format_reader, FormatReader::create(schema_, column_group_->format, cg_file, properties_,
                                                                   needed_columns_, key_retriever_));
    ARROW_ASSIGN_OR_RAISE(auto row_group_in_file, format_reader->get_row_group_infos());
    if (row_group_in_file.empty()) {
      continue;
    }

    size_t rows_in_file = 0;
    if ((cg_file.start_index != 0 || cg_file.end_index != row_group_in_file.back().end_offset)) {
      const auto& start_index = cg_file.start_index;
      const auto& end_index = cg_file.end_index;

      assert(start_index >= 0 && end_index > 0 && start_index < end_index);

      for (size_t j = 0; j < row_group_in_file.size(); ++j) {
        size_t rg_start = row_group_in_file[j].start_offset;
        size_t rg_end = row_group_in_file[j].end_offset;

        // calculate the overlap range
        size_t overlap_start = std::max((size_t)start_index, rg_start);
        size_t overlap_end = std::min((size_t)end_index, rg_end);

        // if the overlap range is valid, create the chunk info
        if (overlap_start < overlap_end) {
          rows_in_file += (overlap_end - overlap_start);
          chunk_infos_.emplace_back(ChunkInfo{
              .file_index = file_idx,
              .row_offset_in_row_group = overlap_start - rg_start,
              .row_offset_in_file = overlap_start,
              .number_of_rows = overlap_end - overlap_start,
              .row_group_index_in_file = j,
              .global_row_end = rows_in_all_files + rows_in_file,
              .avg_memory_size = row_group_in_file[j].memory_size * (overlap_end - overlap_start) / (rg_end - rg_start),
          });
        }
      }
    } else {
      // create the chunk infos with row group indices
      for (size_t j = 0; j < row_group_in_file.size(); ++j) {
        rows_in_file += (row_group_in_file[j].end_offset - row_group_in_file[j].start_offset);
        chunk_infos_.emplace_back(ChunkInfo{
            .file_index = file_idx,
            .row_offset_in_row_group = 0,
            .row_offset_in_file = row_group_in_file[j].start_offset,
            .number_of_rows = row_group_in_file[j].end_offset - row_group_in_file[j].start_offset,
            .row_group_index_in_file = j,
            .global_row_end = rows_in_all_files + rows_in_file,
            .avg_memory_size = row_group_in_file[j].memory_size,
        });
      }
    }

    row_group_infos_.emplace_back(row_group_in_file);
    format_readers_.emplace_back(std::move(format_reader));
    rows_in_all_files += rows_in_file;
  }

  total_rows_ = rows_in_all_files;
  return arrow::Status::OK();
}

size_t ColumnGroupReaderImpl::total_number_of_chunks() const {
  assert(!format_readers_.empty());
  return chunk_infos_.size();
}

size_t ColumnGroupReaderImpl::total_rows() const {
  assert(!format_readers_.empty());
  return total_rows_;
}

arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  assert(!format_readers_.empty());
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
  assert(!format_readers_.empty());
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[chunk_info.file_index]->get_chunk(chunk_info.row_group_index_in_file));

  if (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != rb->num_rows()) {
    rb = rb->Slice(chunk_info.row_offset_in_row_group, chunk_info.number_of_rows);
  }

  return rb;
}

static std::vector<std::vector<int64_t>> split_chunks(const std::vector<int64_t>& sorted_chunk_indices,
                                                      uint64_t parallel_degree) {
  std::vector<std::vector<int64_t>> blocks;
  assert(!sorted_chunk_indices.empty());

#ifndef NDEBUG
  // check sorted, input must be sorted
  for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
    assert(sorted_chunk_indices[i] > sorted_chunk_indices[i - 1]);
  }
#endif

  uint64_t actual_parallel_degree = std::min(parallel_degree, static_cast<uint64_t>(sorted_chunk_indices.size()));

  if (actual_parallel_degree == 0) {
    actual_parallel_degree = 1;
  }

  auto create_continuous_blocks = [&](size_t max_block_size = SIZE_MAX) {
    std::vector<std::vector<int64_t>> continuous_blocks;
    int64_t current_start = sorted_chunk_indices[0];
    int64_t current_count = 1;

    for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
      int64_t next_chunk = sorted_chunk_indices[i];

      if (next_chunk == current_start + current_count && current_count < max_block_size) {
        current_count++;
        continue;
      }
      std::vector<int64_t> block(current_count);
      std::iota(block.begin(), block.end(), current_start);
      continuous_blocks.emplace_back(block);
      current_start = next_chunk;
      current_count = 1;
    }

    if (current_count > 0) {
      std::vector<int64_t> block(current_count);
      std::iota(block.begin(), block.end(), current_start);
      continuous_blocks.emplace_back(block);
    }
    return continuous_blocks;
  };

  if (sorted_chunk_indices.size() <= actual_parallel_degree) {
    return create_continuous_blocks();
  }

  size_t avg_block_size = (sorted_chunk_indices.size() + actual_parallel_degree - 1) / actual_parallel_degree;

  return create_continuous_blocks(avg_block_size);
}

typedef arrow::Result<std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>>> ChunkRBMapResult;

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  assert(!format_readers_.empty());
  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  std::vector<std::future<ChunkRBMapResult>> futures;

  // remove duplicate chunk indices and sort by chunk index
  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  auto folly_thread_pool = ThreadPoolHolder::GetThreadPool(parallelism /* parallelism_hint */);
  auto splitted_chunks = split_chunks(unique_chunk_indices, folly_thread_pool->numThreads());

  auto execute_task = [&format_readers = this->format_readers_, &chunk_infos = this->chunk_infos_](
                          const std::vector<int64_t>& task_indices) -> ChunkRBMapResult {
    std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
    std::vector<std::vector<int64_t>> chunk_idxs_in_files(format_readers.size());

    // Grouping row groups by file
    for (int64_t chunk_index : task_indices) {
      if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos.size())) {
        return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                      std::to_string(chunk_infos.size()));
      }

      const auto& chunk_info = chunk_infos[chunk_index];
      chunk_idxs_in_files[chunk_info.file_index].emplace_back(chunk_index);
    }

    // Read with range and fill chunk_rb_map
    for (size_t file_idx = 0; file_idx < chunk_idxs_in_files.size(); ++file_idx) {
      const auto& chunk_idxs = chunk_idxs_in_files[file_idx];
      if (chunk_idxs.empty()) {
        continue;
      }

      std::vector<std::pair<uint64_t, uint64_t>> ranges_in_file;

      // generate ranges_in_file and combine the range
      for (int64_t chunk_index : chunk_idxs) {
        const auto& chunk_info = chunk_infos[chunk_index];
        if (ranges_in_file.empty()) {
          ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                      chunk_info.row_offset_in_file + chunk_info.number_of_rows);
        } else {
          auto& last_range = ranges_in_file.back();

          // won't be overlay in same file
          assert(chunk_info.row_offset_in_file >= last_range.second);
          if (chunk_info.row_offset_in_file == last_range.second) {
            last_range.second = chunk_info.row_offset_in_file + chunk_info.number_of_rows;
          } else {
            ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                        chunk_info.row_offset_in_file + chunk_info.number_of_rows);
          }
        }
      }

      ARROW_ASSIGN_OR_RAISE(auto reader, format_readers[file_idx]->clone_reader());
      std::vector<std::shared_ptr<arrow::RecordBatch>> rbs_in_file;
      for (auto& range : ranges_in_file) {
        ARROW_ASSIGN_OR_RAISE(auto rbreader, reader->read_with_range(range.first, range.second));
        ARROW_ASSIGN_OR_RAISE(auto rbs, rbreader->ToRecordBatches());
        // append rbs to rbs_in_file
        std::move(rbs.begin(), rbs.end(), std::back_inserter(rbs_in_file));
      }

      // generate chunk_rb_map
      size_t rbs_idx = 0;
      size_t rbs_offset = 0;
      for (size_t i = 0; i < chunk_idxs.size(); ++i) {
        const auto& chunk_info = chunk_infos[chunk_idxs[i]];
        if (UNLIKELY(((rbs_in_file[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows))) {
          return arrow::Status::Invalid("Invalid slice of record batchs: ", std::to_string(chunk_info.number_of_rows),
                                        " out of " + std::to_string(rbs_in_file[rbs_idx]->num_rows() - rbs_offset),
                                        ", [chunk info=", chunk_info.ToString(), "]");
        }

        auto rb = rbs_in_file[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
        chunk_rb_map[chunk_idxs[i]] = rb;
        rbs_offset += chunk_info.number_of_rows;

        assert(rbs_offset <= rbs_in_file[rbs_idx]->num_rows());
        if (rbs_offset == rbs_in_file[rbs_idx]->num_rows()) {
          rbs_idx++;
          rbs_offset = 0;
        }
      }
    }
    return chunk_rb_map;
  };

  for (const auto& task_indices : splitted_chunks) {
    // Capture members by reference to avoid copy and support unique_ptr
    std::packaged_task<ChunkRBMapResult()> task([task_indices, execute_task]() { return execute_task(task_indices); });
    futures.emplace_back(task.get_future());
    folly_thread_pool->add(std::move(task));
  }

  for (auto& future : futures) {
    ARROW_ASSIGN_OR_RAISE(auto res, future.get());
    for (const auto& [k, v] : res) {
      chunk_rb_map.emplace(k, v);
    }
  }

  // 3. generate result
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  for (const auto& chunk_idx : chunk_indices) {
    assert(chunk_rb_map.find(chunk_idx) != chunk_rb_map.end());
    result.emplace_back(chunk_rb_map[chunk_idx]);
  }

  return result;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_size(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].avg_memory_size;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_rows(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  // 111
  return chunk_infos_[chunk_index].number_of_rows;
}

}  // namespace milvus_storage::api
