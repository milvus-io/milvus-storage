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

#include "milvus-storage/reader.h"

#include <cstdio>
#include <mutex>
#include <memory>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/builder.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>
#include <fmt/format.h>
#include <glog/logging.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/thread_pool.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/format_reader_cache.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/column_group_lazy_reader.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/macro.h"

namespace milvus_storage::api {

namespace {

bool metadata_cache_enabled(const Properties& properties) {
  return GetValueNoError<bool>(properties, PROPERTY_READER_METADATA_CACHE_ENABLE);
}

}  // namespace

class PackedRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  /**
   * @brief Open a packed reader to read needed columns in the specified path.
   *
   * @param paths Paths of the packed files to read.
   * @param schema The schema of data to read.
   * @param buffer_size The max buffer size of the packed reader.
   * @param reader_props The reader properties.
   */
  PackedRecordBatchReader(const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
                          MetadataCache metadata_cache,
                          const std::shared_ptr<arrow::Schema>& schema,
                          const std::vector<std::string>& needed_columns,
                          const Properties& properties,
                          const std::function<std::string(const std::string&)>& key_retriever,
                          const std::string& predicate = "")
      : column_groups_(column_groups),
        metadata_cache_(std::move(metadata_cache)),
        schema_(schema),
        needed_columns_(needed_columns),
        out_field_map_(needed_columns.size()),
        out_schema_(nullptr),
        properties_(properties),
        key_retriever_callback_(key_retriever),
        predicate_(predicate),
        number_of_chunks_per_cg_(column_groups.size()),
        chunk_readers_(column_groups.size()),

        current_rbs_(column_groups.size()),
        current_cg_offsets_(column_groups.size(), 0),
        current_cg_chunk_indices_(column_groups.size(), 0),
        loaded_chunk_indices_(column_groups.size()) {
    assert(needed_columns_.size() > 0);
    assert(column_groups.size() > 0);
  }

  /**
   * @brief Open the packed reader by initializing chunk readers for each column group.
   */
  arrow::Status open() {
    // Initialize config from properties
    memory_usage_limit_ =
        milvus_storage::api::GetValueNoError<int64_t>(properties_, PROPERTY_READER_RECORD_BATCH_MAX_SIZE);
    number_of_row_limit_ =
        milvus_storage::api::GetValueNoError<int64_t>(properties_, PROPERTY_READER_RECORD_BATCH_MAX_ROWS);
    parallelism_ = milvus_storage::ThreadPoolHolder::GetParallelism();

    // Predicate pushdown only supported for single column group.
    // Multi-CG filtering requires row-index tracking for cross-CG alignment.
    if (!predicate_.empty() && column_groups_.size() > 1) {
      LOG(WARNING) << "Predicate pushdown is not supported for multi-column-group reads, ignoring predicate";
      predicate_.clear();
    }

    // Create chunk readers for each column group
    bool already_set_total_rows = false;
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(chunk_readers_[i],
                            ColumnGroupReader::create(schema_, column_groups_[i], needed_columns_, properties_,
                                                      key_retriever_callback_, predicate_, metadata_cache_));
      number_of_chunks_per_cg_[i] = chunk_readers_[i]->total_number_of_chunks();
      if (number_of_chunks_per_cg_[i] == 0) {
        return arrow::Status::Invalid(
            fmt::format("Failed to open record batch reader, No chunk to read for column group index: {}", i));
      }

      // check the column group is aligned
      if (!already_set_total_rows) {
        end_of_offset_ = chunk_readers_[i]->total_rows();
        already_set_total_rows = true;
      } else {
        auto cg_rows = chunk_readers_[i]->total_rows();
        // all column groups should have the same number of rows
        if (cg_rows != end_of_offset_) {
          return arrow::Status::Invalid(
              fmt::format("Failed to open record batch reader, Column groups have different number of rows: {} vs {}",
                          end_of_offset_,  // NOLINT
                          cg_rows));
        }
      }
    }
    assert(already_set_total_rows);

    // build the columns after projection in column groups.
    // Must match the column order that ColumnGroupReader produces: it filters
    // needed_columns_ against each CG's columns (see column_group_reader.cpp),
    // so the resulting RecordBatch columns follow needed_columns_ order, not
    // cg->columns (write) order. Iterate needed_columns_ here to keep the
    // (cg_index, idx_in_columns) pairs in sync with rb->column(idx).
    // ex.
    //   needed columns: [F, A, B, C]
    //   column group 0: [A, C, E]
    //   column group 1: [B, D, F]
    //   columns after projection:
    //   column group 0: [A, C]
    //   column group 1: [F, B]
    //
    std::vector<std::vector<std::string_view>> columns_after_projection(column_groups_.size());
    for (int i = 0; i < column_groups_.size(); ++i) {
      const auto& cg = column_groups_[i];
      std::unordered_set<std::string_view> cg_columns_set(cg->columns.begin(), cg->columns.end());
      for (const auto& column : needed_columns_) {
        if (cg_columns_set.count(column) != 0) {
          columns_after_projection[i].emplace_back(column);
        }
      }
    }

    // build column name -> (column group index, index in column group) map
    // ex.
    //  column groups:
    //  column group 0: [A, C]
    //  column group 1: [B, D]
    //  column name map:
    //  A -> (0, 0)
    //  B -> (1, 0)
    //  C -> (0, 1)
    //  D -> (1, 1)
    std::unordered_map<std::string_view, std::pair<int32_t, int32_t>> columnMap;
    for (int i = 0; i < columns_after_projection.size(); ++i) {
      const auto& cgap = columns_after_projection[i];
      for (int j = 0; j < cgap.size(); ++j) {
        columnMap[cgap[j]] = std::make_pair(i, j);
      }
    }

    // When schema is nullptr, derive field types from the column group readers' schemas.
    std::unordered_map<std::string, std::shared_ptr<arrow::Field>> cg_field_map;
    if (!schema_) {
      for (const auto& chunk_reader : chunk_readers_) {
        auto cg_schema = chunk_reader->get_schema();
        if (cg_schema) {
          for (int j = 0; j < cg_schema->num_fields(); ++j) {
            cg_field_map[cg_schema->field(j)->name()] = cg_schema->field(j);
          }
        }
      }
    }

    std::vector<std::shared_ptr<arrow::Field>> out_fields(needed_columns_.size());
    // build output schema and field map
    for (size_t i = 0; i < needed_columns_.size(); ++i) {
      const auto& col_name = needed_columns_[i];
      if (columnMap.count(col_name) == 0) {
        // not found in any column group, missing column should fill with nulls
        if (schema_) {
          auto field_index = schema_->GetFieldIndex(col_name);
          assert(field_index != -1);  // already verified in collect_required_column_groups
        }
        out_field_map_[i] = std::make_pair(-1, -1);
      } else {
        assert(columnMap.find(col_name) != columnMap.end());
        auto [cg_index, idx_in_columns] = columnMap.find(col_name)->second;

        // find the field index in schema
        out_field_map_[i] = std::make_pair(cg_index, idx_in_columns);
      }

      // Get the field from schema or from RecordBatch schema
      if (schema_) {
        out_fields[i] = schema_->field(schema_->GetFieldIndex(col_name));
      } else {
        auto it = cg_field_map.find(col_name);
        if (it != cg_field_map.end()) {
          out_fields[i] = it->second;
        } else {
          return arrow::Status::Invalid(
              fmt::format("Column '{}' not found in any column group and no schema provided", col_name));
        }
      }
    }
    out_schema_ = arrow::schema(out_fields);

    return arrow::Status::OK();
  }

  /**
   * @brief Return the schema of needed columns.
   */
  [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const override { return out_schema_; }

  /**
   * @brief Read next batch of arrow record batch to the specifed pointer.
   *        If the data is drained, return nullptr.
   *
   * @param batch The record batch pointer specified to read.
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out_batch) override {
    // Load data, retrying if predicate filtering drained all rows in a batch
    // but more chunks remain.
    while (true) {
      ARROW_RETURN_NOT_OK(load_internal());

      // EOF check: two paths depending on whether predicate is active.
      // Without predicate: current_offset_ tracks exact row count and matches end_of_offset_.
      // With predicate: filtered rows make current_offset_ < end_of_offset_, so we rely
      // on chunk exhaustion + empty queues to detect EOF.
      if (predicate_.empty()) {
        assert(current_offset_ <= end_of_offset_);
        if (current_offset_ == end_of_offset_) {
          *out_batch = nullptr;
          return arrow::Status::OK();
        }
      }

      // Check if any queue has data
      bool has_data = false;
      for (size_t i = 0; i < column_groups_.size(); ++i) {
        if (!current_rbs_[i].empty()) {
          has_data = true;
          break;
        }
      }
      if (has_data)
        break;

      // Queues empty — check if more chunks to load
      bool more_chunks = false;
      for (size_t i = 0; i < column_groups_.size(); ++i) {
        if (current_cg_chunk_indices_[i] < number_of_chunks_per_cg_[i]) {
          more_chunks = true;
          break;
        }
      }
      if (!more_chunks) {
        *out_batch = nullptr;
        return arrow::Status::OK();
      }
    }

    // begin to callculate the number of rows to return
    size_t min_rows = number_of_row_limit_;
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      assert(!current_rbs_[i].empty());
      auto last_rb = current_rbs_[i].front();

      // we can't concatenate the record batch, so just return the min number of rows
      min_rows = std::min(min_rows, static_cast<size_t>(last_rb->num_rows()));
    }

    // align the record batches from each column group
    // no copy here
    std::vector<std::shared_ptr<arrow::Array>> out_arrays(out_field_map_.size());
    for (int i = 0; i < out_field_map_.size(); ++i) {
      const auto& [out_field_in_cg, idx_in_columns] = out_field_map_[i];

      if (out_field_in_cg == -1) {
        assert(idx_in_columns == -1);
        // fill null column
        ARROW_ASSIGN_OR_RAISE(out_arrays[i], arrow::MakeArrayOfNull(out_schema_->field(i)->type(), min_rows));
        continue;
      }

      auto& rb_queue = current_rbs_[out_field_in_cg];
      assert(!rb_queue.empty());
      auto& rb = rb_queue.front();
      assert(rb->num_rows() >= min_rows);
      if (idx_in_columns >= rb->num_columns()) {
        return arrow::Status::Invalid(
            fmt::format("Column index out of range: {} >= {}", idx_in_columns, rb->num_columns()));
      }

      if (rb->num_rows() == min_rows) {
        // use the whole record batch
        out_arrays[i] = rb->column(idx_in_columns);
      } else {
        // need to slice the record batch
        out_arrays[i] = rb->column(idx_in_columns)->Slice(0, min_rows);
      }
    }

    // update the state
    // rb queues should be updated by popping or slicing
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      auto& rb_queue = current_rbs_[i];
      assert(!rb_queue.empty());
      auto& rb = rb_queue.front();
      if (rb->num_rows() == min_rows) {
        // pop the whole record batch
        rb_queue.pop();
      } else {
        // slice the record batch
        rb_queue.front() = rb->Slice(min_rows);
      }
    }

    current_offset_ += min_rows;

    *out_batch = arrow::RecordBatch::Make(out_schema_, min_rows, out_arrays);

    return arrow::Status::OK();
  }

  private:
  arrow::Result<int64_t> preload_column_group(const size_t& column_group_index, bool& memory_size_unavailable) {
    assert(column_group_index < column_groups_.size());
    std::pair<int32_t, int32_t> result;
    const auto& cg_reader = chunk_readers_[column_group_index];

    auto current_cg_chunk_index = current_cg_chunk_indices_[column_group_index];
    // must have one more chunk to read, should checked in caller
    assert(current_cg_chunk_index < number_of_chunks_per_cg_[column_group_index]);

    auto chunk_size_result = cg_reader->get_chunk_estimated_size(current_cg_chunk_index);
    if (chunk_size_result.ok()) {
      memory_used_ += chunk_size_result.ValueOrDie();
    } else if (chunk_size_result.status().IsNotImplemented()) {
      // Memory estimation is optional. Keep normal scans usable, but stop
      // scheduling extra chunks after every column group has one chunk queued.
      memory_size_unavailable = true;
    } else {
      return chunk_size_result.status();
    }
    ARROW_ASSIGN_OR_RAISE(auto chunk_rows, cg_reader->get_chunk_rows(current_cg_chunk_index));

    // update the states after loading column group info
    // will update current_rbs_ queue after preload
    current_cg_offsets_[column_group_index] += chunk_rows;
    current_cg_chunk_indices_[column_group_index] = current_cg_chunk_index + 1;

    return current_cg_chunk_index;
  }

  arrow::Status load_internal() {
    // Only load when ANY column group's queue is empty and has more chunks
    bool need_load = false;
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      if (current_rbs_[i].empty() && current_cg_chunk_indices_[i] < number_of_chunks_per_cg_[i]) {
        need_load = true;
        break;
      }
    }

    if (!need_load) {
      return arrow::Status::OK();
    }

    // Reset memory tracking for this batch
    memory_used_ = 0;

    // Load chunks up to memory limit using min-heap for row offset alignment.
    // Each column group is guaranteed at least one chunk per load round,
    // regardless of memory limit, to avoid leaving any CG with an empty queue.
    RowOffsetMinHeap sorted_offsets;
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      if (current_cg_chunk_indices_[i] < number_of_chunks_per_cg_[i]) {
        sorted_offsets.emplace(i, current_cg_offsets_[i]);
      }
    }

    auto all_cgs_have_data = [&]() {
      for (size_t i = 0; i < column_groups_.size(); ++i) {
        if (current_rbs_[i].empty() && loaded_chunk_indices_[i].empty() &&
            current_cg_chunk_indices_[i] < number_of_chunks_per_cg_[i]) {
          return false;
        }
      }
      return true;
    };

    bool memory_size_unavailable = false;
    while (!sorted_offsets.empty() &&
           ((!memory_size_unavailable && memory_used_ < memory_usage_limit_) || !all_cgs_have_data())) {
      auto [cg_index, cg_offset] = sorted_offsets.top();
      sorted_offsets.pop();

      if (current_cg_chunk_indices_[cg_index] >= number_of_chunks_per_cg_[cg_index]) {
        continue;
      }

      ARROW_ASSIGN_OR_RAISE(auto loaded_chunk_idx, preload_column_group(cg_index, memory_size_unavailable));
      loaded_chunk_indices_[cg_index].emplace_back(loaded_chunk_idx);

      // Push back to heap if more chunks available
      if (current_cg_chunk_indices_[cg_index] < number_of_chunks_per_cg_[cg_index]) {
        sorted_offsets.emplace(cg_index, current_cg_offsets_[cg_index]);
      }
    }

    // Execute I/O
    for (size_t cg_idx = 0; cg_idx < loaded_chunk_indices_.size(); ++cg_idx) {
      if (loaded_chunk_indices_[cg_idx].empty()) {
        continue;
      }

      if (!predicate_.empty()) {
        // Predicate path: read chunks individually via get_chunk() to avoid
        // the chunk-slicing in get_chunks()/read_chunks_from_files(), which
        // assumes pre-filter row counts.
        for (auto chunk_idx : loaded_chunk_indices_[cg_idx]) {
          ARROW_ASSIGN_OR_RAISE(auto rb, chunk_readers_[cg_idx]->get_chunk(chunk_idx));
          if (rb && rb->num_rows() > 0) {
            current_rbs_[cg_idx].push(rb);
          }
        }
      } else {
        ARROW_ASSIGN_OR_RAISE(auto record_batchs,
                              chunk_readers_[cg_idx]->get_chunks(loaded_chunk_indices_[cg_idx], parallelism_));
        for (const auto& record_batch : record_batchs) {
          assert(record_batch);
          current_rbs_[cg_idx].push(record_batch);
        }
      }

      loaded_chunk_indices_[cg_idx].clear();
    }

    return arrow::Status::OK();
  }

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  MetadataCache metadata_cache_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> needed_columns_;
  std::vector<std::pair<std::int32_t, std::int32_t>> out_field_map_;
  std::shared_ptr<arrow::Schema> out_schema_;
  Properties properties_;
  std::function<std::string(const std::string&)> key_retriever_callback_;
  std::string predicate_;

  size_t end_of_offset_;                         // end offset of the data to read
  std::vector<size_t> number_of_chunks_per_cg_;  // total number of chunks for each column group
  std::vector<std::unique_ptr<ColumnGroupReader>> chunk_readers_;

  int64_t number_of_row_limit_;
  int64_t memory_usage_limit_;
  size_t parallelism_;
  // above is immutable after open

  int64_t memory_used_{0};     // current memory usage
  int64_t current_offset_{0};  // current read offset
  std::vector<std::queue<std::shared_ptr<arrow::RecordBatch>>>
      current_rbs_;                                // current read arrays for each column group
  std::vector<size_t> current_cg_offsets_;         // current read offset for each column group
  std::vector<int64_t> current_cg_chunk_indices_;  // current chunk index for each column group

  std::vector<std::vector<int64_t>> loaded_chunk_indices_;  // use to cache loaded chunk indices
};

// ==================== ChunkReaderImpl Implementation ====================

/**
 * @brief Concrete implementation of ChunkReader interface
 */
class ChunkReaderImpl : public ChunkReader {
  public:
  /**
   * @brief Constructs a ChunkReaderImpl for a specific column group
   *
   * @param schema Shared pointer to the schema of the dataset
   * @param column_group Shared pointer to the column group metadata and configuration
   * @param needed_columns Subset of columns to read (empty = all columns)
   * @param properties Read properties including encryption settings
   *
   * @throws std::invalid_argument if fs or column_group is null
   */
  ChunkReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                  const std::shared_ptr<ColumnGroup>& column_group,
                  const std::vector<std::string>& needed_columns,
                  const Properties& properties,
                  const std::function<std::string(const std::string&)>& key_retriever,
                  MetadataCache metadata_cache);

  // open the reader
  arrow::Status open();
  folly::SemiFuture<arrow::Status> open_async();

  /**
   * @brief Destructor
   */
  ~ChunkReaderImpl() override = default;

  // Implement ChunkReader interface
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism) override;
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> get_chunks_async(
      const std::vector<int64_t>& chunk_indices, size_t parallelism) override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_estimated_size() override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_column_estimated_size(
      const std::string& field_name) override;
  [[nodiscard]] arrow::Result<std::vector<std::vector<uint64_t>>> get_chunk_column_estimated_size() override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_rows() override;

  private:
  [[nodiscard]] arrow::Status validate_needed_columns() const;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks_sync(
      const std::vector<int64_t>& chunk_indices, size_t parallelism);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<ColumnGroup> column_group_;  ///< Column group metadata and configuration
  std::vector<std::string> needed_columns_;    ///< Subset of columns to read (empty = all columns)
  Properties properties_;
  std::function<std::string(const std::string&)> key_retriever_callback_;
  MetadataCache metadata_cache_;
  std::unique_ptr<ColumnGroupReader> chunk_reader_;
};

// ==================== ChunkReaderImpl Method Implementations ====================

ChunkReaderImpl::ChunkReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                 const std::shared_ptr<ColumnGroup>& column_group,
                                 const std::vector<std::string>& needed_columns,
                                 const Properties& properties,
                                 const std::function<std::string(const std::string&)>& key_retriever,
                                 MetadataCache metadata_cache)
    : schema_(schema),
      column_group_(column_group),
      needed_columns_(needed_columns),
      properties_(properties),
      key_retriever_callback_(key_retriever),
      metadata_cache_(std::move(metadata_cache)) {}

arrow::Status ChunkReaderImpl::validate_needed_columns() const {
  // The case is:
  // column groups:
  //   - group1 :[a, b]
  //   - group2 :[c, d]
  // needed columns: [c, d]
  //
  // But open the group1 which not exist the needed columns
  std::unordered_set<std::string_view> columns_set;
  for (const auto& col_name : column_group_->columns) {
    columns_set.insert(col_name);
  }

  bool exist_in_group = false;
  for (const auto& col_name : needed_columns_) {
    if (columns_set.find(col_name) != columns_set.end()) {
      exist_in_group = true;
      break;
    }
  }

  if (!exist_in_group) {
    // TODO(jiaqizho): more info in invalid message
    return arrow::Status::Invalid("No needed columns found in column group");
  }
  return arrow::Status::OK();
}

arrow::Status ChunkReaderImpl::open() {
  ARROW_RETURN_NOT_OK(validate_needed_columns());

  ARROW_ASSIGN_OR_RAISE(chunk_reader_, ColumnGroupReader::create(schema_, column_group_, needed_columns_, properties_,
                                                                 key_retriever_callback_, "", metadata_cache_));
  return arrow::Status::OK();
}

folly::SemiFuture<arrow::Status> ChunkReaderImpl::open_async() {
  auto validation_status = validate_needed_columns();
  if (!validation_status.ok()) {
    return folly::makeSemiFuture(validation_status);
  }

  // Publish chunk_reader_ only after the column-group reader has completed its
  // file metadata initialization.
  return ColumnGroupReader::create_async(schema_, column_group_, needed_columns_, properties_, key_retriever_callback_,
                                         "", metadata_cache_)
      .deferValue([this](arrow::Result<std::unique_ptr<ColumnGroupReader>>&& reader_result) -> arrow::Status {
        ARROW_ASSIGN_OR_RAISE(chunk_reader_, std::move(reader_result));
        return arrow::Status::OK();
      });
}

size_t ChunkReaderImpl::total_number_of_chunks() const { return chunk_reader_->total_number_of_chunks(); }

arrow::Result<std::vector<int64_t>> ChunkReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  return chunk_reader_->get_chunk_indices(row_indices);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ChunkReaderImpl::get_chunk(int64_t chunk_index) {
  return chunk_reader_->get_chunk(chunk_index);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  return get_chunks_sync(chunk_indices, parallelism);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReaderImpl::get_chunks_sync(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> results;
  ARROW_ASSIGN_OR_RAISE(results, chunk_reader_->get_chunks(chunk_indices, parallelism));

  // no need to slice
  if (results.size() == chunk_indices.size()) {
    return results;
  }

  // slice to match the original chunk indices
  std::vector<std::shared_ptr<arrow::RecordBatch>> sliced_results;
  sliced_results.reserve(chunk_indices.size());

  size_t curr_rb_index = 0;
  size_t curr_rb_offset = 0;
  for (long long target_chunk_index : chunk_indices) {
    ARROW_ASSIGN_OR_RAISE(auto target_number_of_rows, chunk_reader_->get_chunk_rows(target_chunk_index));

    if (curr_rb_index >= results.size()) {
      return arrow::Status::Invalid(
          fmt::format("Failed to slice the record batch, Invalid result from chunk_reader, the row size of result not "
                      "match. [target_chunk_index={}]",
                      target_chunk_index));
    }

    auto rb = results[curr_rb_index];
    if (UNLIKELY(curr_rb_offset + target_number_of_rows > rb->num_rows())) {
      return arrow::Status::Invalid(fmt::format(
          "Failed to slice the record batch, current record batch is discontinuous. "
          "[target_chunk_index={}] [curr_rb_index={}] [curr_rb_offset={}] [target_number_of_rows={}] [rb_num_rows={}]",
          target_chunk_index,     // NOLINT
          curr_rb_index,          // NOLINT
          curr_rb_offset,         // NOLINT
          target_number_of_rows,  // NOLINT
          rb->num_rows()));
    }

    if (curr_rb_offset == 0 && target_number_of_rows == rb->num_rows()) {
      sliced_results.emplace_back(rb);
    } else {
      sliced_results.emplace_back(rb->Slice(curr_rb_offset, target_number_of_rows));
    }

    curr_rb_offset += target_number_of_rows;
    if (curr_rb_offset == rb->num_rows()) {
      curr_rb_offset = 0;
      curr_rb_index++;
    }
  }

  assert(sliced_results.size() == chunk_indices.size());
  return sliced_results;
}

folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> ChunkReaderImpl::get_chunks_async(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  // Plan on sorted unique chunks; fan-in below restores caller order and duplicates.
  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  auto total_chunks = chunk_reader_->total_number_of_chunks();
  for (auto idx : unique_chunk_indices) {
    if (UNLIKELY(idx < 0 || static_cast<size_t>(idx) >= total_chunks)) {
      return folly::makeSemiFuture(arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(
          arrow::Status::Invalid(fmt::format("Chunk index out of range: {} out of {}", idx, total_chunks))));
    }
  }
  if (unique_chunk_indices.empty()) {
    return folly::makeSemiFuture(arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(
        std::vector<std::shared_ptr<arrow::RecordBatch>>{}));
  }

  auto get_chunk_info = [this](int64_t chunk_index) -> const ChunkInfo& {
    return chunk_reader_->get_chunk_info(chunk_index);
  };
  // Natural tasks preserve contiguous file ranges. Optional splitting changes
  // task granularity only; it does not select or cap an executor.
  auto all_tasks = ChunkTask::Build(unique_chunk_indices, get_chunk_info);
  SplitAsyncTasks(all_tasks, std::max<size_t>(parallelism, 1), ChunkTask::SplitTraits{get_chunk_info},
                  GetAsyncTaskSplitStrategy(properties_));

  std::vector<folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>> futures;
  std::vector<std::vector<int64_t>> task_chunk_lists;
  futures.reserve(all_tasks.size());
  task_chunk_lists.reserve(all_tasks.size());

  // Create every backend future before fan-in. parallelism influences task
  // splitting, but this layer does not impose a separate in-flight limit.
  for (auto& task : all_tasks) {
    task_chunk_lists.push_back(task.chunk_indices);
    futures.push_back(chunk_reader_->get_chunks_async(task));
  }

  // Associate results with task chunk ids, then rebuild the original request;
  // asynchronous completion order is intentionally irrelevant.
  return folly::collectAll(std::move(futures))
      .deferValue([chunk_indices, task_chunk_lists = std::move(task_chunk_lists)](
                      auto&& all_results) mutable -> arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> {
        std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> all_rbs;
        for (size_t i = 0; i < all_results.size(); ++i) {
          auto& tryResult = all_results[i];
          if (tryResult.hasException()) {
            return arrow::Status::IOError(tryResult.exception().what().toStdString());
          }
          ARROW_ASSIGN_OR_RAISE(auto rbs, std::move(tryResult.value()));

          auto& chunk_list = task_chunk_lists[i];
          if (rbs.size() != chunk_list.size()) {
            return arrow::Status::Invalid(
                fmt::format("Async chunk task returned {} batches for {} chunks", rbs.size(), chunk_list.size()));
          }
          for (size_t j = 0; j < rbs.size(); ++j) {
            all_rbs[chunk_list[j]] = std::move(rbs[j]);
          }
        }

        std::vector<std::shared_ptr<arrow::RecordBatch>> result;
        result.reserve(chunk_indices.size());
        for (auto idx : chunk_indices) {
          auto it = all_rbs.find(idx);
          if (it == all_rbs.end()) {
            return arrow::Status::Invalid(fmt::format("Missing async chunk result for chunk {}", idx));
          }
          result.push_back(it->second);
        }
        return result;
      });
}

arrow::Result<std::vector<uint64_t>> ChunkReaderImpl::get_chunk_estimated_size() {
  const auto total_chunks = total_number_of_chunks();
  std::vector<uint64_t> result(total_chunks);
  assert(total_chunks > 0);

  for (size_t i = 0; i < total_chunks; ++i) {
    ARROW_ASSIGN_OR_RAISE(result[i], chunk_reader_->get_chunk_estimated_size(i));
  }

  return result;
}

arrow::Result<std::vector<uint64_t>> ChunkReaderImpl::get_chunk_column_estimated_size(const std::string& field_name) {
  const auto field = std::find(column_group_->columns.begin(), column_group_->columns.end(), field_name);
  if (field == column_group_->columns.end()) {
    return arrow::Status::Invalid(fmt::format("Column '{}' is not part of the column group", field_name));
  }
  if (std::find(std::next(field), column_group_->columns.end(), field_name) != column_group_->columns.end()) {
    return arrow::Status::Invalid(fmt::format("Column '{}' is duplicated in the column group", field_name));
  }
  const auto col_idx = static_cast<int>(std::distance(column_group_->columns.begin(), field));

  const auto total_chunks = total_number_of_chunks();
  std::vector<uint64_t> result(total_chunks);
  for (size_t i = 0; i < total_chunks; ++i) {
    ARROW_ASSIGN_OR_RAISE(result[i], chunk_reader_->get_chunk_column_estimated_size(i, col_idx));
  }
  return result;
}

arrow::Result<std::vector<std::vector<uint64_t>>> ChunkReaderImpl::get_chunk_column_estimated_size() {
  const auto total_chunks = total_number_of_chunks();
  std::vector<std::vector<uint64_t>> result(column_group_->columns.size(), std::vector<uint64_t>(total_chunks));
  for (size_t col_idx = 0; col_idx < column_group_->columns.size(); ++col_idx) {
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
      ARROW_ASSIGN_OR_RAISE(result[col_idx][chunk_idx],
                            chunk_reader_->get_chunk_column_estimated_size(chunk_idx, static_cast<int>(col_idx)));
    }
  }
  return result;
}

arrow::Result<std::vector<uint64_t>> ChunkReaderImpl::get_chunk_rows() {
  const auto total_chunks = total_number_of_chunks();
  std::vector<uint64_t> result(total_chunks);
  assert(total_chunks > 0);

  for (size_t i = 0; i < total_chunks; ++i) {
    ARROW_ASSIGN_OR_RAISE(result[i], chunk_reader_->get_chunk_rows(i));
  }

  return result;
}

// ==================== ReaderImpl Implementation ====================

/**
 * @brief Concrete implementation of the Reader interface
 *
 * This class provides the actual implementation for reading data from milvus
 * storage datasets using manifest-based metadata. It supports efficient batch
 * reading, column projection, filtering, and parallel processing of large datasets
 * stored in packed columnar format.
 */
class ReaderImpl : public Reader {
  public:
  /**
   * @brief Constructs a ReaderImpl instance for a milvus storage dataset
   *
   * Initializes the reader with dataset column groups and configuration. The
   * column groups provides metadata about column groups, data layout, and storage
   * locations, enabling optimized query planning and execution.
   *
   * @param cgs Dataset column group information
   * @param schema Arrow schema defining the logical structure of the data
   * @param needed_columns Optional vector of column names to read (nullptr reads all columns)
   * @param properties Read configuration properties including encryption settings
   */
  ReaderImpl(const std::shared_ptr<ColumnGroups>& cgs,
             const std::shared_ptr<arrow::Schema>& schema,
             const std::shared_ptr<std::vector<std::string>>& needed_columns,
             const Properties& properties)
      : cgs_(cgs),
        schema_(schema),
        needed_columns_(needed_columns),
        properties_(properties),
        key_retriever_callback_(nullptr),
        metadata_cache_(metadata_cache_enabled(properties)) {
    // Validate required parameters
    assert(cgs_);
  }

  [[nodiscard]] std::shared_ptr<ColumnGroups> get_column_groups() const override {
    assert(cgs_);
    return cgs_;
  }

  /**
   * @brief Performs a full table scan with optional filtering and buffering
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& predicate) const override {
    // empty column groups
    if (cgs_->size() == 0) {
      if (schema_) {
        ARROW_ASSIGN_OR_RAISE(auto empty_table, arrow::Table::MakeEmpty(schema_));
        return std::make_shared<arrow::TableBatchReader>(std::move(empty_table));
      }

      return arrow::Status::Invalid("Cannot read from empty column groups without a schema");
    }

    ARROW_ASSIGN_OR_RAISE(auto resolved_columns, resolve_needed_columns(schema_, needed_columns_));

    // Collect required column groups and share the ReaderImpl-owned metadata cache.
    auto needed_column_group_indices = collect_required_column_group_indices(resolved_columns);
    ARROW_ASSIGN_OR_RAISE(auto needed_column_groups, column_groups_from_indices(needed_column_group_indices));
    auto metadata_cache = get_metadata_cache();

    // Build projected schema: from user-provided schema or nullptr
    std::shared_ptr<arrow::Schema> projected_schema = nullptr;
    if (schema_) {
      std::vector<std::shared_ptr<arrow::Field>> needed_fields;
      for (const auto& column_name : resolved_columns) {
        auto field = schema_->GetFieldByName(column_name);
        if (field != nullptr) {
          needed_fields.emplace_back(field);
        }
      }
      projected_schema = arrow::schema(needed_fields);
    }

    auto reader =
        std::make_shared<PackedRecordBatchReader>(needed_column_groups, std::move(metadata_cache), projected_schema,
                                                  resolved_columns, properties_, key_retriever_callback_, predicate);
    ARROW_RETURN_NOT_OK(reader->open());
    return reader;
  }

  /**
   * @brief Get a chunk reader for a specific column group
   */
  [[nodiscard]] arrow::Result<std::unique_ptr<ChunkReader>> get_chunk_reader(
      int64_t column_group_index,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) const override {
    if (column_group_index < 0 || static_cast<size_t>(column_group_index) >= cgs_->size()) {
      return arrow::Status::Invalid(
          fmt::format("Failed to get chunk reader, column group index out of range: {} (size: {})",
                      column_group_index,  // NOLINT
                      cgs_->size()));
    }
    auto column_group = (*cgs_)[column_group_index];
    if (!column_group) {
      return arrow::Status::Invalid(
          fmt::format("Failed to get chunk reader, column group at index {} is null", column_group_index));
    }
    ARROW_ASSIGN_OR_RAISE(auto resolved_columns,
                          resolve_needed_columns(schema_, effective_needed_columns(needed_columns)));

    auto metadata_cache = get_metadata_cache();
    auto chunk_reader = std::make_unique<ChunkReaderImpl>(schema_, column_group, resolved_columns, properties_,
                                                          key_retriever_callback_, std::move(metadata_cache));
    ARROW_RETURN_NOT_OK(chunk_reader->open());
    return chunk_reader;
  }

  [[nodiscard]] folly::SemiFuture<arrow::Result<std::unique_ptr<ChunkReader>>> get_chunk_reader_async(
      int64_t column_group_index,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) const override {
    if (column_group_index < 0 || static_cast<size_t>(column_group_index) >= cgs_->size()) {
      return folly::makeSemiFuture(arrow::Result<std::unique_ptr<ChunkReader>>(arrow::Status::Invalid(
          fmt::format("Failed to get chunk reader, column group index out of range: {} (size: {})",
                      column_group_index,  // NOLINT
                      cgs_->size()))));
    }
    auto column_group = (*cgs_)[column_group_index];
    if (!column_group) {
      return folly::makeSemiFuture(arrow::Result<std::unique_ptr<ChunkReader>>(arrow::Status::Invalid(
          fmt::format("Failed to get chunk reader, column group at index {} is null", column_group_index))));
    }

    FOLLY_ARROW_ASSIGN_OR_RAISE(auto resolved_columns,
                                resolve_needed_columns(schema_, effective_needed_columns(needed_columns)));

    auto metadata_cache = get_metadata_cache();
    auto chunk_reader = std::make_unique<ChunkReaderImpl>(schema_, column_group, resolved_columns, properties_,
                                                          key_retriever_callback_, std::move(metadata_cache));
    auto* chunk_reader_ptr = chunk_reader.get();
    // The continuation owns the ChunkReaderImpl while open_async() initializes
    // it through the temporary raw pointer.
    return chunk_reader_ptr->open_async().deferValue(
        [chunk_reader =
             std::move(chunk_reader)](arrow::Status status) mutable -> arrow::Result<std::unique_ptr<ChunkReader>> {
          ARROW_RETURN_NOT_OK(status);
          return std::move(chunk_reader);
        });
  }

  /**
   * @brief Extracts specific rows by their global indices with parallel processing
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(
      const std::vector<int64_t>& row_indices,
      size_t parallelism = 1,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) override {
    // empty input row indices
    if (row_indices.empty()) {
      if (schema_) {
        return arrow::Table::MakeEmpty(schema_);
      }
      return arrow::Status::Invalid("Cannot create empty table without a schema");
    }

    // empty column groups
    if (cgs_->empty()) {
      return arrow::Status::Invalid("Empty column groups without empty input row indices");
    }

    ARROW_ASSIGN_OR_RAISE(auto resolved_columns,
                          resolve_needed_columns(schema_, effective_needed_columns(needed_columns)));
    auto needed_column_group_indices = collect_required_column_group_indices(resolved_columns);
    ARROW_ASSIGN_OR_RAISE(auto needed_column_groups, column_groups_from_indices(needed_column_group_indices));
    ARROW_ASSIGN_OR_RAISE(auto lazy_readers, create_lazy_readers(needed_column_groups, resolved_columns));

    std::vector<std::shared_ptr<arrow::Table>> tables;
    ARROW_ASSIGN_OR_RAISE(tables, take_tables_sync(row_indices, *lazy_readers, parallelism));

    return build_take_table(tables, row_indices, resolved_columns, schema_);
  }

  [[nodiscard]] folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(
      const std::vector<int64_t>& row_indices,
      size_t parallelism = 1,
      const std::shared_ptr<std::vector<std::string>>& needed_columns = nullptr) override {
    if (row_indices.empty()) {
      if (!schema_) {
        return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
            arrow::Status::Invalid("Cannot create empty table without a schema")));
      }
      FOLLY_ARROW_ASSIGN_OR_RAISE(auto empty_table, arrow::Table::MakeEmpty(schema_));
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(std::move(empty_table)));
    }

    if (cgs_->empty()) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
          arrow::Status::Invalid("Empty column groups without empty input row indices")));
    }

    FOLLY_ARROW_ASSIGN_OR_RAISE(auto resolved_columns,
                                resolve_needed_columns(schema_, effective_needed_columns(needed_columns)));

    auto needed_column_group_indices = collect_required_column_group_indices(resolved_columns);
    FOLLY_ARROW_ASSIGN_OR_RAISE(auto needed_column_groups, column_groups_from_indices(needed_column_group_indices));
    FOLLY_ARROW_ASSIGN_OR_RAISE(auto lazy_readers, create_lazy_readers(needed_column_groups, resolved_columns));

    // Fan out file-aware tasks across all required column groups, then combine
    // the reordered per-group tables into the requested logical column order.
    return take_tables_async(row_indices, needed_column_groups, lazy_readers, parallelism)
        .deferValue([row_indices, resolved_columns = std::move(resolved_columns), schema = schema_,
                     lazy_readers](auto&& tables_result) mutable -> arrow::Result<std::shared_ptr<arrow::Table>> {
          ARROW_ASSIGN_OR_RAISE(auto tables, std::move(tables_result));
          return build_take_table(tables, row_indices, resolved_columns, schema);
        });
  }

  private:
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::Table>>> take_tables_sync(
      const std::vector<int64_t>& row_indices,
      std::vector<std::unique_ptr<ColumnGroupLazyReader>>& lazy_readers,
      size_t parallelism);
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::Table>>>> take_tables_async(
      const std::vector<int64_t>& row_indices,
      const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
      std::shared_ptr<std::vector<std::unique_ptr<ColumnGroupLazyReader>>> lazy_readers,
      size_t parallelism);

  [[nodiscard]] arrow::Result<std::shared_ptr<std::vector<std::unique_ptr<ColumnGroupLazyReader>>>> create_lazy_readers(
      const std::vector<std::shared_ptr<ColumnGroup>>& needed_column_groups,
      const std::vector<std::string>& resolved_columns) const;

  [[nodiscard]] static arrow::Result<std::shared_ptr<arrow::Table>> build_take_table(
      const std::vector<std::shared_ptr<arrow::Table>>& tables,
      const std::vector<int64_t>& row_indices,
      const std::vector<std::string>& resolved_columns,
      const std::shared_ptr<arrow::Schema>& schema);

  std::shared_ptr<ColumnGroups> cgs_;                         ///< Dataset column groups with metadata and layout info
  std::shared_ptr<arrow::Schema> schema_;                     ///< Logical Arrow schema defining data structure
  std::shared_ptr<std::vector<std::string>> needed_columns_;  ///< Column projection (nullptr = all columns)
  Properties properties_;                                     ///< Configuration properties including encryption
  std::function<std::string(const std::string&)>
      key_retriever_callback_;  ///< Callback function for retrieving encryption keys
  mutable std::mutex metadata_cache_mutex_;
  MetadataCache metadata_cache_;

  /**
   * @brief Returns the per-call needed_columns if non-null and non-empty, otherwise falls back to the default.
   */
  [[nodiscard]] const std::shared_ptr<std::vector<std::string>>& effective_needed_columns(
      const std::shared_ptr<std::vector<std::string>>& per_call) const {
    return (per_call != nullptr && !per_call->empty()) ? per_call : needed_columns_;
  }

  /**
   * @brief Resolve needed columns and verify they exist in schema.
   *
   * If needed_columns is nullptr or empty:
   *   - When schema is provided, returns all schema field names.
   *   - When schema is nullptr, returns all column names from column groups.
   * Otherwise returns the provided columns after verifying each exists in the schema (if schema is provided).
   */
  arrow::Result<std::vector<std::string>> resolve_needed_columns(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<std::vector<std::string>>& needed_columns) const {
    // Caller-supplied projections must have distinct names. Duplicates would
    // propagate into filtered_columns / needed_column_indices_ and confuse
    // PackedRecordBatchReader's (cg_index, idx_in_rb) bookkeeping
    // (columnMap keeps only the last occurrence) as well as Arrow's Parquet
    // reader (duplicate column indices). Reject early at the single gateway
    // covered by every entry point (get_record_batch_reader / get_chunk_reader
    // / take, including per-call overrides).
    if (needed_columns != nullptr && !needed_columns->empty()) {
      std::unordered_set<std::string_view> seen;
      seen.reserve(needed_columns->size());
      for (const auto& name : *needed_columns) {
        if (!seen.insert(name).second) {
          return arrow::Status::Invalid(fmt::format("needed_columns contains duplicate column name: '{}'", name));
        }
      }
    }

    std::vector<std::string> resolved;
    if (needed_columns != nullptr && !needed_columns->empty()) {
      resolved = *needed_columns;
    } else if (schema) {
      resolved.reserve(schema->num_fields());
      for (int i = 0; i < schema->num_fields(); ++i) {
        resolved.emplace_back(schema->field(i)->name());
      }
    } else {
      // No schema provided: collect all unique column names from column groups
      std::unordered_set<std::string> seen;
      for (const auto& cg : *cgs_) {
        if (!cg) {
          continue;
        }
        for (const auto& col : cg->columns) {
          if (seen.insert(col).second) {
            resolved.emplace_back(col);
          }
        }
      }
    }
    if (schema) {
      for (const auto& col_name : resolved) {
        if (schema->GetFieldIndex(col_name) == -1) {
          return arrow::Status::Invalid("Column [name=", col_name, "] not found in schema");
        }
      }
    }
    return resolved;
  }

  /**
   * @brief Collects unique column groups for the requested columns
   */
  [[nodiscard]] std::vector<std::shared_ptr<ColumnGroup>> collect_required_column_groups(
      const std::vector<std::string>& needed_columns) const {
    std::unordered_set<std::shared_ptr<ColumnGroup>> unique_groups;

    for (const auto& column_name : needed_columns) {
      auto column_group = std::find_if(cgs_->begin(), cgs_->end(), [&column_name](const auto& cg) {
        return std::find(cg->columns.begin(), cg->columns.end(), column_name) != cg->columns.end();
      });
      if (column_group == cgs_->end()) {
        continue;  // Skip missing column groups
      }
      unique_groups.insert(*column_group);
    }

    // The case is:
    // 1. Schema with a,b,c three fields
    // 2. Column groups with a,b two fields
    // 3. Projection is c which can't find the column group
    if (unique_groups.empty() && cgs_->size() > 0) {
      // use the first column group
      unique_groups.insert((*cgs_)[0]);
    }

    return {unique_groups.begin(), unique_groups.end()};
  }

  [[nodiscard]] std::vector<size_t> collect_required_column_group_indices(
      const std::vector<std::string>& needed_columns) const {
    std::vector<size_t> indices;
    std::unordered_set<size_t> seen;

    for (const auto& column_name : needed_columns) {
      auto column_group = std::find_if(cgs_->begin(), cgs_->end(), [&column_name](const auto& cg) {
        return cg && std::find(cg->columns.begin(), cg->columns.end(), column_name) != cg->columns.end();
      });
      if (column_group == cgs_->end()) {
        continue;
      }

      auto index = static_cast<size_t>(std::distance(cgs_->begin(), column_group));
      if (seen.insert(index).second) {
        indices.emplace_back(index);
      }
    }

    // Preserve the existing behavior: when every requested column is missing
    // from storage column groups, read the first group so callers can still
    // infer row count and fill projected missing columns with nulls.
    if (indices.empty() && cgs_->size() > 0) {
      indices.emplace_back(0);
    }

    return indices;
  }

  arrow::Result<std::vector<std::shared_ptr<ColumnGroup>>> column_groups_from_indices(
      const std::vector<size_t>& indices) const {
    std::vector<std::shared_ptr<ColumnGroup>> column_groups;
    column_groups.reserve(indices.size());
    for (auto index : indices) {
      if (index >= cgs_->size()) {
        return arrow::Status::Invalid(
            fmt::format("Column group index out of range: {} (size: {})", index, cgs_->size()));
      }
      auto column_group = (*cgs_)[index];
      if (!column_group) {
        return arrow::Status::Invalid(fmt::format("Column group at index {} is null", index));
      }
      column_groups.emplace_back(std::move(column_group));
    }
    return column_groups;
  }

  MetadataCache get_metadata_cache() const {
    std::lock_guard<std::mutex> lock(metadata_cache_mutex_);
    return metadata_cache_;
  }

  void set_keyretriever(const std::function<std::string(const std::string&)>& callback) override {
    std::lock_guard<std::mutex> lock(metadata_cache_mutex_);
    metadata_cache_ = MetadataCache(metadata_cache_enabled(properties_));
    key_retriever_callback_ = callback;
  }
};

arrow::Result<std::shared_ptr<std::vector<std::unique_ptr<ColumnGroupLazyReader>>>> ReaderImpl::create_lazy_readers(
    const std::vector<std::shared_ptr<ColumnGroup>>& needed_column_groups,
    const std::vector<std::string>& resolved_columns) const {
  // Shared ownership keeps the complete reader set alive across take_async()
  // fan-out and its final continuation.
  auto lazy_readers =
      std::make_shared<std::vector<std::unique_ptr<ColumnGroupLazyReader>>>(needed_column_groups.size());
  auto metadata_cache = get_metadata_cache();
  for (size_t i = 0; i < needed_column_groups.size(); ++i) {
    if (!needed_column_groups[i]) {
      return arrow::Status::Invalid(fmt::format("Failed to call take, column group at index {} is empty", i));
    }
    ARROW_ASSIGN_OR_RAISE((*lazy_readers)[i],
                          ColumnGroupLazyReader::create(schema_, needed_column_groups[i], properties_, resolved_columns,
                                                        key_retriever_callback_, metadata_cache));
  }
  return lazy_readers;
}

arrow::Result<std::shared_ptr<arrow::Table>> ReaderImpl::build_take_table(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    const std::vector<int64_t>& row_indices,
    const std::vector<std::string>& resolved_columns,
    const std::shared_ptr<arrow::Schema>& schema) {
  // Every input table represents one column group and has already been reordered
  // to the requested row order. Flatten their columns after verifying row counts.
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Field>> out_fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> out_arrays;
  std::unordered_map<std::string_view, size_t> colname_to_index;

  uint64_t last_row_counts = UINT64_MAX;
  for (const auto& table : tables) {
    assert(table);
    if (last_row_counts == UINT64_MAX) {
      last_row_counts = table->num_rows();
    } else if (last_row_counts != table->num_rows()) {
      return arrow::Status::Invalid("Logical error, different row counts in column groups");
    }

    const auto& table_schema = table->schema();
    for (int i = 0; i < table->num_columns(); ++i) {
      colname_to_index[table_schema->field(i)->name()] = fields.size();
      fields.emplace_back(table_schema->field(i));
      columns.emplace_back(table->column(i));
    }
  }
  assert(fields.size() == columns.size());

  // Rebuild the caller's logical projection order. Columns absent from storage
  // are synthesized as nulls only when their type is available from schema.
  out_arrays.reserve(resolved_columns.size());
  for (const auto& colname : resolved_columns) {
    auto it = colname_to_index.find(colname);
    if (it == colname_to_index.end()) {
      if (!schema) {
        return arrow::Status::Invalid(
            fmt::format("Column '{}' not found in any column group and no schema provided for null filling", colname));
      }
      auto missing_field = schema->GetFieldByName(colname);
      ARROW_ASSIGN_OR_RAISE(auto null_array,
                            arrow::MakeArrayOfNull(missing_field->type(), static_cast<int64_t>(row_indices.size())));
      out_fields.emplace_back(missing_field);
      out_arrays.emplace_back(
          std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{std::move(null_array)}, missing_field->type()));
    } else {
      out_fields.emplace_back(fields[it->second]);
      out_arrays.emplace_back(columns[it->second]);
    }
  }

  // All arrays now share the requested row count and logical column order.
  return arrow::Table::Make(arrow::schema(out_fields), out_arrays, static_cast<int64_t>(row_indices.size()));
}

arrow::Result<std::vector<std::shared_ptr<arrow::Table>>> ReaderImpl::take_tables_sync(
    const std::vector<int64_t>& row_indices,
    std::vector<std::unique_ptr<ColumnGroupLazyReader>>& lazy_readers,
    size_t parallelism) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  tables.reserve(lazy_readers.size());
  for (const auto& lazy_reader : lazy_readers) {
    ARROW_ASSIGN_OR_RAISE(auto table, lazy_reader->take(row_indices, parallelism));
    tables.emplace_back(std::move(table));
  }
  return tables;
}

folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::Table>>>> ReaderImpl::take_tables_async(
    const std::vector<int64_t>& row_indices,
    const std::vector<std::shared_ptr<ColumnGroup>>& column_groups,
    std::shared_ptr<std::vector<std::unique_ptr<ColumnGroupLazyReader>>> lazy_readers,
    size_t parallelism) {
  // Build one flat task list across all required column groups and files before fan-out.
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto all_tasks, TakeTask::Build(column_groups, row_indices));
  auto& readers = *lazy_readers;

  SplitAsyncTasks(all_tasks, std::max<size_t>(parallelism, 1), TakeTask::SplitTraits{},
                  GetAsyncTaskSplitStrategy(properties_));

  std::vector<size_t> task_cg_indices;
  std::vector<std::vector<size_t>> task_positions;
  std::vector<folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>>> futures;
  task_cg_indices.reserve(all_tasks.size());
  task_positions.reserve(all_tasks.size());
  futures.reserve(all_tasks.size());

  // Submit the complete cross-column-group task set before waiting, allowing
  // native async backends to overlap work across both files and column groups.
  for (auto& task : all_tasks) {
    task_cg_indices.push_back(task.reader_index);
    task_positions.push_back(std::move(task.original_positions));
    futures.push_back(readers[task.reader_index]->take_async(task));
  }

  // Keep the readers alive for every in-flight task. The saved positions remove
  // both task-splitting order and completion order from the final row order.
  return folly::collectAll(std::move(futures))
      .deferValue([row_indices, lazy_readers, task_cg_indices = std::move(task_cg_indices),
                   task_positions = std::move(task_positions)](
                      auto&& all_results) mutable -> arrow::Result<std::vector<std::shared_ptr<arrow::Table>>> {
        std::vector<std::vector<std::shared_ptr<arrow::Table>>> per_cg_tables(lazy_readers->size());
        std::vector<std::vector<size_t>> per_cg_positions(lazy_readers->size());

        for (size_t i = 0; i < all_results.size(); ++i) {
          auto& tryResult = all_results[i];
          if (tryResult.hasException()) {
            return arrow::Status::IOError(tryResult.exception().what().toStdString());
          }
          ARROW_ASSIGN_OR_RAISE(auto table, std::move(tryResult.value()));
          size_t cg_idx = task_cg_indices[i];
          per_cg_tables[cg_idx].push_back(std::move(table));
          per_cg_positions[cg_idx].insert(per_cg_positions[cg_idx].end(), task_positions[i].begin(),
                                          task_positions[i].end());
        }

        std::vector<std::shared_ptr<arrow::Table>> tables;
        tables.reserve(lazy_readers->size());
        for (size_t cg = 0; cg < lazy_readers->size(); ++cg) {
          if (per_cg_tables[cg].empty()) {
            continue;
          }
          ARROW_ASSIGN_OR_RAISE(auto concatenated, arrow::ConcatenateTables(per_cg_tables[cg]));

          auto& positions = per_cg_positions[cg];
          if (positions.size() != row_indices.size()) {
            return arrow::Status::Invalid(fmt::format("Async take returned {} row positions for {} requested rows",
                                                      positions.size(), row_indices.size()));
          }
          std::vector<int64_t> reorder(row_indices.size());
          for (size_t i = 0; i < positions.size(); ++i) {
            reorder[positions[i]] = static_cast<int64_t>(i);
          }
          ARROW_ASSIGN_OR_RAISE(auto reordered, CopySelectedRows(concatenated, reorder));
          tables.push_back(std::move(reordered));
        }
        return tables;
      });
}

// ==================== Factory Function Implementation ====================

std::unique_ptr<Reader> Reader::create(const std::shared_ptr<ColumnGroups>& cgs,
                                       const std::shared_ptr<arrow::Schema>& schema,
                                       const std::shared_ptr<std::vector<std::string>>& needed_columns,
                                       const Properties& properties) {
  return std::make_unique<ReaderImpl>(cgs, schema, needed_columns, properties);
}

}  // namespace milvus_storage::api
