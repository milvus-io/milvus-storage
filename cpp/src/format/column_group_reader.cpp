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
#include <fmt/format.h>

#include <folly/executors/IOThreadPoolExecutor.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage::api {

using milvus_storage::RowGroupInfo;
using ChunkRBMapResult = arrow::Result<std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>>>;
struct ChunkInfo {
  public:
  size_t file_index;               // current chunk belong which file
  size_t row_offset_in_row_group;  // the starting row offset of this row group in its file
  size_t row_offset_in_file;       // the starting row offset of file
  size_t number_of_rows;           // number of rows in this row group
  size_t row_group_index_in_file;  // the index of this row group in its file
  size_t global_row_end;           // the ending row offset of this row group in the whole chunk reader
  size_t avg_memory_size;          // average memory usage of this row group

  [[nodiscard]] std::string ToString() const;
};

template <typename ReaderT>
class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever,
                        const milvus_storage::MetadataCache& cache,
                        const std::string& predicate = "");

  ~ColumnGroupReaderImpl() override = default;

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] size_t total_rows() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_size(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_column_sizes(int64_t chunk_index) override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  ChunkRBMapResult read_chunks_from_files(const std::vector<int64_t>& task_indices);
  arrow::Result<std::shared_ptr<ReaderT>> open_reader_for_file(size_t file_index);

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  milvus_storage::MetadataCache cache_;
  std::string predicate_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  std::shared_ptr<arrow::Schema> file_schema_;
  size_t total_rows_ = 0;

  std::vector<std::shared_ptr<ReaderT>> format_readers_;
  bool opened_ = false;
};  // ColumnGroupReaderImpl

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

template <typename ReaderT>
arrow::Status ColumnGroupReaderImpl<ReaderT>::open() {
  const auto& cg_files = column_group_->files;

  // init chunk infos
  size_t rows_in_all_files = 0;
  row_group_infos_.clear();
  row_group_infos_.resize(cg_files.size());
  file_schema_.reset();
  format_readers_.clear();
  format_readers_.resize(cg_files.size());
  chunk_infos_.clear();

  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto& cg_file = cg_files[file_idx];

    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", file_idx, cg_file.path));
    }

    std::vector<RowGroupInfo> row_group_in_file;
    if (cache_.enabled()) {
      auto key = ReaderT::MetaTrait::cache_key(cg_file);
      ARROW_ASSIGN_OR_RAISE(auto metadata, cache_.get<ReaderT>()->get_or_open(key, [this, cg_file]() {
        return FormatReader::load_metadata<ReaderT>(cg_file, properties_, key_retriever_);
      }));
      row_group_in_file = metadata->row_group_infos;
      if (!file_schema_ && metadata->file_schema) {
        file_schema_ = metadata->file_schema;
      }
    } else {
      ARROW_ASSIGN_OR_RAISE(format_readers_[file_idx], open_reader_for_file(file_idx));
      ARROW_ASSIGN_OR_RAISE(row_group_in_file, format_readers_[file_idx]->get_row_group_infos());
      if (!file_schema_) {
        file_schema_ = format_readers_[file_idx]->get_schema();
      }
    }
    row_group_infos_[file_idx] = row_group_in_file;
    if (row_group_in_file.empty()) {
      continue;
    }

    if (cache_.enabled()) {
      ARROW_ASSIGN_OR_RAISE(format_readers_[file_idx], open_reader_for_file(file_idx));
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
        size_t overlap_start = std::max(static_cast<size_t>(start_index), rg_start);
        size_t overlap_end = std::min(static_cast<size_t>(end_index), rg_end);

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

    rows_in_all_files += rows_in_file;
  }

  total_rows_ = rows_in_all_files;
  opened_ = true;
  return arrow::Status::OK();
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<ReaderT>> ColumnGroupReaderImpl<ReaderT>::open_reader_for_file(size_t file_index) {
  if (file_index >= column_group_->files.size()) {
    return arrow::Status::Invalid("Column group file index out of range: ", file_index,
                                  " >= ", column_group_->files.size());
  }

  auto file = column_group_->files[file_index];
  if (!cache_.enabled()) {
    ARROW_ASSIGN_OR_RAISE(auto reader, FormatReader::create(schema_, column_group_->format, file, properties_,
                                                            needed_columns_, key_retriever_));
    if (!predicate_.empty()) {
      ARROW_RETURN_NOT_OK(reader->set_predicate(predicate_));
    }
    auto typed_reader = std::dynamic_pointer_cast<ReaderT>(reader);
    if (!typed_reader) {
      return arrow::Status::Invalid("FormatReader::create returned incompatible reader for format: ",
                                    column_group_->format);
    }
    return typed_reader;
  } else {
    auto key = ReaderT::MetaTrait::cache_key(file);
    ARROW_ASSIGN_OR_RAISE(auto metadata, cache_.get<ReaderT>()->get_or_open(key, [this, file]() {
      return FormatReader::load_metadata<ReaderT>(file, properties_, key_retriever_);
    }));
    return FormatReader::create_from_metadata<ReaderT>(metadata, file, schema_, needed_columns_, predicate_);
  }

  return arrow::Status::Invalid("Unreachable code");
}

template <typename ReaderT>
size_t ColumnGroupReaderImpl<ReaderT>::total_number_of_chunks() const {
  return chunk_infos_.size();
}

template <typename ReaderT>
size_t ColumnGroupReaderImpl<ReaderT>::total_rows() const {
  return total_rows_;
}

template <typename ReaderT>
arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl<ReaderT>::get_chunk_indices(
    const std::vector<int64_t>& row_indices) {
  assert(opened_);
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(chunk_infos_.begin(), chunk_infos_.end(), row_index,
                               [](int64_t a, const ChunkInfo& b) { return a < b.global_row_end; });
    auto chunk_index = std::distance(chunk_infos_.begin(), it);
    if (chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid(fmt::format("Row index out of range: {}", row_index));
    }

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl<ReaderT>::get_chunk(int64_t chunk_index) {
  assert(opened_);
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  if (!format_readers_[chunk_info.file_index]) {
    ARROW_ASSIGN_OR_RAISE(format_readers_[chunk_info.file_index], open_reader_for_file(chunk_info.file_index));
  }
  ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[chunk_info.file_index]->get_chunk(chunk_info.row_group_index_in_file));

  // With predicate, Vortex's WithRowRange + WithFilter already produced the
  // correct subset — skip slicing since filtered row counts don't match
  // pre-filter chunk metadata.
  if (predicate_.empty() && (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != rb->num_rows())) {
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

template <typename ReaderT>
ChunkRBMapResult ColumnGroupReaderImpl<ReaderT>::read_chunks_from_files(const std::vector<int64_t>& task_indices) {
  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  std::vector<std::vector<int64_t>> chunk_idxs_in_files(column_group_->files.size());

  // Grouping row groups by file
  for (int64_t chunk_index : task_indices) {
    if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
      return arrow::Status::Invalid(
          fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
    }

    const auto& chunk_info = chunk_infos_[chunk_index];
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
      const auto& chunk_info = chunk_infos_[chunk_index];
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

    ARROW_ASSIGN_OR_RAISE(auto reader, open_reader_for_file(file_idx));
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
    for (long long chunk_idx : chunk_idxs) {
      const auto& chunk_info = chunk_infos_[chunk_idx];
      if (UNLIKELY(((rbs_in_file[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows))) {
        return arrow::Status::Invalid(
            fmt::format("Invalid slice of record batchs: {} out of {}, [chunk info={}]", chunk_info.number_of_rows,
                        rbs_in_file[rbs_idx]->num_rows() - rbs_offset, chunk_info.ToString()));
      }

      auto rb = rbs_in_file[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
      chunk_rb_map[chunk_idx] = rb;
      rbs_offset += chunk_info.number_of_rows;

      assert(rbs_offset <= rbs_in_file[rbs_idx]->num_rows());
      if (rbs_offset == rbs_in_file[rbs_idx]->num_rows()) {
        rbs_idx++;
        rbs_offset = 0;
      }
    }
  }
  return chunk_rb_map;
}

template <typename ReaderT>
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl<ReaderT>::get_chunks(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  assert(opened_);

  // inject fault
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));

  // remove duplicate chunk indices and sort by chunk index
  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;

  if (parallelism <= 1) {
    ARROW_ASSIGN_OR_RAISE(chunk_rb_map, read_chunks_from_files(unique_chunk_indices));
  } else {
    auto folly_thread_pool = ThreadPoolHolder::GetThreadPool(parallelism /* parallelism_hint */);
    auto splitted_chunks = split_chunks(unique_chunk_indices, folly_thread_pool->numThreads());
    std::vector<std::future<ChunkRBMapResult>> futures;

    for (const auto& task_indices : splitted_chunks) {
      std::packaged_task<ChunkRBMapResult()> task(
          [this, task_indices]() { return read_chunks_from_files(task_indices); });
      futures.emplace_back(task.get_future());
      folly_thread_pool->add(std::move(task));
    }

    // Wait for all futures to complete before checking errors,
    // to avoid early return while tasks still hold `this`.
    std::vector<ChunkRBMapResult> all_results;
    all_results.reserve(futures.size());
    for (auto& future : futures) {
      all_results.emplace_back(future.get());
    }
    for (auto& result : all_results) {
      ARROW_ASSIGN_OR_RAISE(auto res, std::move(result));
      for (const auto& [k, v] : res) {
        chunk_rb_map.emplace(k, v);
      }
    }
  }

  // generate result
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  for (const auto& chunk_idx : chunk_indices) {
    assert(chunk_rb_map.find(chunk_idx) != chunk_rb_map.end());
    result.emplace_back(chunk_rb_map[chunk_idx]);
  }

  return result;
}

template <typename ReaderT>
arrow::Result<uint64_t> ColumnGroupReaderImpl<ReaderT>::get_chunk_size(int64_t chunk_index) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].avg_memory_size;
}

template <typename ReaderT>
arrow::Result<std::vector<uint64_t>> ColumnGroupReaderImpl<ReaderT>::get_chunk_column_sizes(int64_t chunk_index) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }

  const auto& chunk_info = chunk_infos_[chunk_index];
  if (!format_readers_[chunk_info.file_index]) {
    ARROW_ASSIGN_OR_RAISE(format_readers_[chunk_info.file_index], open_reader_for_file(chunk_info.file_index));
  }

  // Raw per-projected-column footer weights (unnormalized) for this chunk's row group.
  // Empty means the backend has no weights; we then split the chunk memory evenly.
  ARROW_ASSIGN_OR_RAISE(auto weights, format_readers_[chunk_info.file_index]->get_column_sizes(
                                          static_cast<int>(chunk_info.row_group_index_in_file)));

  // The chunk's real in-memory size (already slice-adjusted for partial row groups; this is
  // exactly what get_chunk_size(chunk_index) returns). We distribute it across columns so the
  // per-column values sum to M exactly, absorbing any integer-division remainder.
  const uint64_t M = chunk_info.avg_memory_size;

  // Number of projected columns: the weight vector length when available, otherwise the
  // reader's projected column count. Projection is needed_columns_ (already filtered to the
  // group's columns); an empty projection means all columns in this column group.
  size_t num_cols = weights.size();
  if (num_cols == 0) {
    if (!needed_columns_.empty()) {
      num_cols = needed_columns_.size();
    } else if (column_group_) {
      num_cols = column_group_->columns.size();
    }
  }
  if (num_cols == 0) {
    return std::vector<uint64_t>{};
  }

  std::vector<uint64_t> result(num_cols, 0);

  uint64_t weight_sum = 0;
  if (weights.size() == num_cols) {
    for (const auto w : weights) {
      weight_sum += w;
    }
  }

  if (weight_sum > 0) {
    // Proportional split by weight, then hand the floor-rounding remainder to the largest
    // weights so the total lands exactly on M.
    uint64_t assigned = 0;
    for (size_t i = 0; i < num_cols; ++i) {
      result[i] = static_cast<uint64_t>((static_cast<__uint128_t>(M) * weights[i]) / weight_sum);
      assigned += result[i];
    }
    uint64_t remainder = M - assigned;  // assigned <= M since each floor(M*w/sum) <= M*w/sum
    // Distribute the remainder 1 byte at a time to the largest-weight columns first.
    std::vector<size_t> order(num_cols);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    for (size_t k = 0; remainder > 0 && !order.empty(); ++k) {
      result[order[k % order.size()]] += 1;
      --remainder;
    }
  } else {
    // Even split: base share to every column, then spread the remainder one byte each.
    const uint64_t base = M / num_cols;
    uint64_t remainder = M - base * num_cols;
    for (size_t i = 0; i < num_cols; ++i) {
      result[i] = base + (i < remainder ? 1 : 0);
    }
  }

  return result;
}

template <typename ReaderT>
arrow::Result<uint64_t> ColumnGroupReaderImpl<ReaderT>::get_chunk_rows(int64_t chunk_index) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].number_of_rows;
}

template <typename ReaderT>
std::shared_ptr<arrow::Schema> ColumnGroupReaderImpl<ReaderT>::get_schema() const {
  return file_schema_;
}

template <typename ReaderT>
ColumnGroupReaderImpl<ReaderT>::ColumnGroupReaderImpl(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<api::ColumnGroup>& column_group,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
    const milvus_storage::MetadataCache& cache,
    const std::string& predicate)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      cache_(cache),
      predicate_(predicate) {}

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    const std::string& predicate,
    const milvus_storage::MetadataCache& cache) {
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }
  const bool cache_enabled =
      cache.enabled() && GetValueNoError<bool>(properties, PROPERTY_READER_METADATA_CACHE_ENABLE);

  // Generate the output schema with only the needed columns
  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
    }
  }

  if (schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& col_name : filtered_columns) {
      auto field = schema->GetFieldByName(col_name);
      if (!field) {
        return arrow::Status::Invalid(
            "ColumnGroupReader: column '" + col_name +
            "' found in column_group but not in schema. Schema fields: " + schema->ToString());
      }
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }

  auto create_reader = [&](const milvus_storage::MetadataCache& metadata_cache) {
    return metadata_cache.dispatch(
        column_group->format, [&](auto typed_cache) -> arrow::Result<std::unique_ptr<ColumnGroupReader>> {
          if (!typed_cache) {
            return arrow::Status::Invalid("Format reader metadata cache is null");
          }

          using TypedCache = typename decltype(typed_cache)::element_type;
          using ReaderT = typename TypedCache::ReaderType;
          std::unique_ptr<ColumnGroupReader> reader = std::make_unique<ColumnGroupReaderImpl<ReaderT>>(
              out_schema, column_group, properties, filtered_columns, key_retriever, metadata_cache, predicate);
          ARROW_RETURN_NOT_OK(reader->open());
          return reader;
        });
  };

  if (!cache_enabled) {
    return create_reader(milvus_storage::MetadataCache(false));
  }

  return create_reader(cache);
}

}  // namespace milvus_storage::api
