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

#include <numeric>
#include <algorithm>
#include <vector>
#include <algorithm>

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/iterator.h>
#include <parquet/properties.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/column_group_lazy_reader.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/common/macro.h"

namespace milvus_storage::api {

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
                          const std::shared_ptr<arrow::Schema>& schema,
                          const std::vector<std::string>& needed_columns,
                          const Properties& properties,
                          const std::function<std::string(const std::string&)>& key_retriever)
      : column_groups_(column_groups),
        schema_(schema),
        needed_columns_(needed_columns),
        out_field_map_(needed_columns.size()),
        out_schema_(nullptr),
        properties_(properties),
        key_retriever_callback_(key_retriever),
        number_of_chunks_per_cg_(column_groups.size()),
        chunk_readers_(column_groups.size()),
        // above is immutable after open
        memory_used_(0),
        current_offset_(0),
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

    // Create chunk readers for each column group
    bool already_set_total_rows = false;
    for (size_t i = 0; i < column_groups_.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(chunk_readers_[i], ColumnGroupReader::create(schema_, column_groups_[i], needed_columns_,
                                                                         properties_, key_retriever_callback_));
      number_of_chunks_per_cg_[i] = chunk_readers_[i]->total_number_of_chunks();
      if (number_of_chunks_per_cg_[i] == 0) {
        return arrow::Status::Invalid("No chunk to read for column group index: " + std::to_string(i));
      }

      // check the column group is aligned
      if (!already_set_total_rows) {
        end_of_offset_ = chunk_readers_[i]->total_rows();
        already_set_total_rows = true;
      } else {
        auto cg_rows = chunk_readers_[i]->total_rows();
        // all column groups should have the same number of rows
        if (cg_rows != end_of_offset_) {
          return arrow::Status::Invalid("Column groups have different number of rows: " +
                                        std::to_string(end_of_offset_) + " vs " + std::to_string(cg_rows));
        }
      }
    }
    assert(already_set_total_rows);

    // build the columns after projection in column groups
    // ex.
    //   needed columns: [A, B, C, F]
    //   column group 0: [A, C, E]
    //   column group 1: [B, D, F]
    //   columns after projection:
    //   column group 0: [A, C]
    //   column group 1: [B, F]
    //
    std::vector<std::vector<std::string_view>> columns_after_projection(column_groups_.size());
    {
      std::unordered_set<std::string_view> unique_needed_columns(needed_columns_.begin(), needed_columns_.end());

      for (int i = 0; i < column_groups_.size(); ++i) {
        const auto& cg = column_groups_[i];
        for (int j = 0; j < cg->columns.size(); ++j) {
          if (unique_needed_columns.count(cg->columns[j]) != 0) {
            columns_after_projection[i].emplace_back(cg->columns[j]);
          }
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

    std::vector<std::shared_ptr<arrow::Field>> out_fields(needed_columns_.size());
    // build output schema and field map
    for (size_t i = 0; i < needed_columns_.size(); ++i) {
      const auto& col_name = needed_columns_[i];
      if (columnMap.count(col_name) == 0) {
        // no found in any column group, missing column should fill with nulls(maybe use the default value?)
        auto field_index = schema_->GetFieldIndex(col_name);
        if (field_index == -1) {
          // should not happen, already checked the projection outer reader
          return arrow::Status::Invalid("Needed column " + col_name + " not found in schema");
        }

        out_field_map_[i] = std::make_pair(-1, -1);
      } else {
        assert(columnMap.find(col_name) != columnMap.end());
        auto [cg_index, idx_in_columns] = columnMap.find(col_name)->second;

        // find the field index in schema
        out_field_map_[i] = std::make_pair(cg_index, idx_in_columns);
      }
      out_fields[i] = schema_->field(schema_->GetFieldIndex(col_name));
    }
    out_schema_ = arrow::schema(out_fields);

    return arrow::Status::OK();
  }

  /**
   * @brief Return the schema of needed columns.
   */
  std::shared_ptr<arrow::Schema> schema() const override { return out_schema_; }

  /**
   * @brief Read next batch of arrow record batch to the specifed pointer.
   *        If the data is drained, return nullptr.
   *
   * @param batch The record batch pointer specified to read.
   */
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out_batch) override {
    // load data into buffer until reaching memory limit or all data loaded
    ARROW_RETURN_NOT_OK(load_internal());

    assert(current_offset_ <= end_of_offset_);

    // end of data
    if (current_offset_ == end_of_offset_) {
      *out_batch = nullptr;
      return arrow::Status::OK();
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

        // only chunk been poped out will release memory
        ARROW_ASSIGN_OR_RAISE(auto chunk_idxs, chunk_readers_[i]->get_chunk_indices({current_offset_}));
        assert(chunk_idxs.size() == 1);
        ARROW_ASSIGN_OR_RAISE(auto pop_chunk_size, chunk_readers_[i]->get_chunk_size(chunk_idxs[0]));
        memory_used_ -= pop_chunk_size;
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
  arrow::Result<int64_t> preload_column_group(const size_t& column_group_index) {
    assert(column_group_index < column_groups_.size());
    std::pair<int32_t, int32_t> result;
    const auto& cg_reader = chunk_readers_[column_group_index];

    auto current_cg_chunk_index = current_cg_chunk_indices_[column_group_index];
    // must have one more chunk to read, should checked in caller
    assert(current_cg_chunk_index < number_of_chunks_per_cg_[column_group_index]);

    ARROW_ASSIGN_OR_RAISE(auto chunk_size, cg_reader->get_chunk_size(current_cg_chunk_index));
    ARROW_ASSIGN_OR_RAISE(auto chunk_rows, cg_reader->get_chunk_rows(current_cg_chunk_index));

    // update the states after loading column group info
    // will update current_rbs_ queue after preload
    memory_used_ += chunk_size;
    current_cg_offsets_[column_group_index] += chunk_rows;
    current_cg_chunk_indices_[column_group_index] = current_cg_chunk_index + 1;

    return current_cg_chunk_index;
  }

  arrow::Status load_internal(size_t min_rows_in_memory = 1) {
    // load one chunk from each of the column groups
    for (size_t column_group_index = 0; column_group_index < column_groups_.size(); ++column_group_index) {
      auto current_cg_chunk_index = current_cg_chunk_indices_[column_group_index];
      assert(current_cg_chunk_index <= number_of_chunks_per_cg_[column_group_index]);
      if (current_cg_chunk_index == number_of_chunks_per_cg_[column_group_index]) {
        // no more chunk in current column group
        continue;
      }

      if (current_cg_offsets_[column_group_index] - current_offset_ >= min_rows_in_memory) {
        // do have enough rows in memory
        continue;
      }
#ifndef NDEBUG
      else {
        auto current_cg_remain_rows = current_cg_offsets_[column_group_index] - current_offset_;
        auto rows_in_rb_queue = 0;
        std::queue<std::shared_ptr<arrow::RecordBatch>> rb_queue_copy = current_rbs_[column_group_index];
        while (!rb_queue_copy.empty()) {
          auto rb = rb_queue_copy.front();
          rows_in_rb_queue += rb->num_rows();
          rb_queue_copy.pop();
        }

        assert(rows_in_rb_queue == current_cg_remain_rows);
      }
#endif

      ARROW_ASSIGN_OR_RAISE(auto loaded_chunk_idx, preload_column_group(column_group_index));
      loaded_chunk_indices_[column_group_index].emplace_back(loaded_chunk_idx);
    }

    // still have memory, try to load more chunks
    if (memory_used_ < memory_usage_limit_) {
      RowOffsetMinHeap sorted_offsets;
      for (size_t i = 0; i < column_groups_.size(); ++i) {
        sorted_offsets.emplace(i, current_cg_offsets_[i]);
      }

      // find the smallest rows of column group to load
      while (!sorted_offsets.empty() && memory_used_ < memory_usage_limit_) {
        auto [cg_index, cg_offset] = sorted_offsets.top();
        sorted_offsets.pop();

        auto current_cg_chunk_index = current_cg_chunk_indices_[cg_index];
        assert(current_cg_chunk_index <= number_of_chunks_per_cg_[cg_index]);
        if (current_cg_chunk_index == number_of_chunks_per_cg_[cg_index]) {
          // no more chunk in current column group
          continue;
        }

        ARROW_ASSIGN_OR_RAISE(auto loaded_chunk_idx, preload_column_group(cg_index));
        loaded_chunk_indices_[cg_index].emplace_back(loaded_chunk_idx);
        // push back to heap
        sorted_offsets.emplace(cg_index, current_cg_offsets_[cg_index]);
      }
    }

#ifndef NDEBUG
    // loaded_chunk_indices_ should be sorted
    for (size_t i = 0; i < loaded_chunk_indices_.size(); ++i) {
      auto& chunk_index = loaded_chunk_indices_[i];
      for (size_t j = 1; j < chunk_index.size(); ++j) {
        assert(chunk_index[j] > chunk_index[j - 1]);
      }
    }
#endif

    // finally, load the chunks
    for (size_t column_group_index = 0; column_group_index < loaded_chunk_indices_.size(); ++column_group_index) {
      if (loaded_chunk_indices_[column_group_index].empty()) {
        continue;
      }

      ARROW_ASSIGN_OR_RAISE(auto record_batchs, chunk_readers_[column_group_index]->get_chunks(
                                                    loaded_chunk_indices_[column_group_index], 1 /* parallelism */));

      // push to the queue
      for (const auto& record_batch : record_batchs) {
        assert(record_batch);
        current_rbs_[column_group_index].push(record_batch);
      }

      // always clearup the temp loaded info
      loaded_chunk_indices_[column_group_index].clear();
    }

    return arrow::Status::OK();
  }

  private:
  std::vector<std::shared_ptr<ColumnGroup>> column_groups_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> needed_columns_;
  std::vector<std::pair<std::int32_t, std::int32_t>> out_field_map_;
  std::shared_ptr<arrow::Schema> out_schema_;
  Properties properties_;
  std::function<std::string(const std::string&)> key_retriever_callback_;

  size_t end_of_offset_;                         // end offset of the data to read
  std::vector<size_t> number_of_chunks_per_cg_;  // total number of chunks for each column group
  std::vector<std::unique_ptr<ColumnGroupReader>> chunk_readers_;

  int64_t number_of_row_limit_;
  int64_t memory_usage_limit_;
  // above is immutable after open

  int64_t memory_used_;     // current memory usage
  int64_t current_offset_;  // current read offset
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
                  const std::function<std::string(const std::string&)>& key_retriever);

  // open the reader
  arrow::Status open();

  /**
   * @brief Destructor
   */
  ~ChunkReaderImpl() override = default;

  // Implement ChunkReader interface
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_size() override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_chunk_rows() override;

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<ColumnGroup> column_group_;  ///< Column group metadata and configuration
  std::vector<std::string> needed_columns_;    ///< Subset of columns to read (empty = all columns)
  Properties properties_;
  std::function<std::string(const std::string&)> key_retriever_callback_;
  std::unique_ptr<ColumnGroupReader> chunk_reader_;
};

// ==================== ChunkReaderImpl Method Implementations ====================

ChunkReaderImpl::ChunkReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                 const std::shared_ptr<ColumnGroup>& column_group,
                                 const std::vector<std::string>& needed_columns,
                                 const Properties& properties,
                                 const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      needed_columns_(needed_columns),
      properties_(properties),
      key_retriever_callback_(key_retriever) {
  // create schema from column group
  assert(schema != nullptr);
}

arrow::Status ChunkReaderImpl::open() {
  ARROW_ASSIGN_OR_RAISE(chunk_reader_, ColumnGroupReader::create(schema_, column_group_, needed_columns_, properties_,
                                                                 key_retriever_callback_));

  return arrow::Status::OK();
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
  for (size_t i = 0; i < chunk_indices.size(); ++i) {
    int64_t target_chunk_index = chunk_indices[i];
    ARROW_ASSIGN_OR_RAISE(auto target_number_of_rows, chunk_reader_->get_chunk_rows(target_chunk_index));

    if (curr_rb_index >= results.size()) {
      return arrow::Status::Invalid(
          "Invalid result from chunk_reader, the row size of result not match. [target_chunk_index=" +
          std::to_string(target_chunk_index) + "]");
    }

    auto rb = results[curr_rb_index];
    if (UNLIKELY(curr_rb_offset + target_number_of_rows > rb->num_rows())) {
      return arrow::Status::Invalid(
          "Failed to slice the record batch, current record batch is discontinuous. [target_chunk_index=",
          std::to_string(target_chunk_index), "] [curr_rb_index=", std::to_string(curr_rb_index),
          "] [curr_rb_offset=", std::to_string(curr_rb_offset),
          "] [target_number_of_rows=", std::to_string(target_number_of_rows),
          "] [rb_num_rows=", std::to_string(rb->num_rows()), "]");
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

arrow::Result<std::vector<uint64_t>> ChunkReaderImpl::get_chunk_size() {
  const auto total_chunks = total_number_of_chunks();
  std::vector<uint64_t> result(total_chunks);
  assert(total_chunks > 0);

  for (size_t i = 0; i < total_chunks; ++i) {
    ARROW_ASSIGN_OR_RAISE(result[i], chunk_reader_->get_chunk_size(i));
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
      : cgs_(cgs), schema_(schema), properties_(properties), key_retriever_callback_(nullptr) {
    // Validate required parameters
    assert(cgs_ && schema_);

    // Initialize the list of columns to read from the dataset
    if (needed_columns != nullptr && !needed_columns->empty()) {
      needed_columns_ = *needed_columns;
    } else {
      // If no specific columns requested, read all columns from the schema
      needed_columns_.clear();
      needed_columns_.reserve(schema_->num_fields());
      for (size_t i = 0; i < schema_->num_fields(); ++i) {
        needed_columns_.emplace_back(schema_->field(i)->name());
      }
    }
    assert(!needed_columns_.empty());
  }

  std::shared_ptr<ColumnGroups> get_column_groups() const override {
    assert(cgs_);
    return cgs_;
  }

  /**
   * @brief Performs a full table scan with optional filtering and buffering
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> get_record_batch_reader(
      const std::string& /*predicate*/) const override {
    // empty column groups
    if (cgs_->size() == 0) {
      // direct return empty record batch reader
      ARROW_ASSIGN_OR_RAISE(auto empty_table, arrow::Table::MakeEmpty(schema_));
      return std::make_shared<arrow::TableBatchReader>(std::move(empty_table));
    }

    // Collect required column groups if not already done
    ARROW_RETURN_NOT_OK(collect_required_column_groups());

    // Create schema with only needed columns for projection
    std::vector<std::shared_ptr<arrow::Field>> needed_fields;
    for (const auto& column_name : needed_columns_) {
      auto field = schema_->GetFieldByName(column_name);
      if (field != nullptr) {
        needed_fields.emplace_back(field);
      }
    }
    auto projected_schema = arrow::schema(needed_fields);

    auto reader = std::make_shared<PackedRecordBatchReader>(needed_column_groups_, projected_schema, needed_columns_,
                                                            properties_, key_retriever_callback_);
    ARROW_RETURN_NOT_OK(reader->open());
    return reader;
  }

  /**
   * @brief Get a chunk reader for a specific column group
   */
  [[nodiscard]] arrow::Result<std::unique_ptr<ChunkReader>> get_chunk_reader(
      int64_t column_group_index) const override {
    if (column_group_index < 0 || static_cast<size_t>(column_group_index) >= cgs_->size()) {
      return arrow::Status::Invalid("Column group index out of range: " + std::to_string(column_group_index) +
                                    " (size: " + std::to_string(cgs_->size()) + ")");
    }
    auto column_group = (*cgs_)[column_group_index];
    if (!column_group) {
      return arrow::Status::Invalid("Column group at index " + std::to_string(column_group_index) + " is null");
    }

    auto chunk_reader =
        std::make_unique<ChunkReaderImpl>(schema_, column_group, needed_columns_, properties_, key_retriever_callback_);
    ARROW_RETURN_NOT_OK(chunk_reader->open());
    return chunk_reader;
  }

  /**
   * @brief Extracts specific rows by their global indices with parallel processing
   */
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices,
                                                                  size_t parallelism) override {
    ARROW_RETURN_NOT_OK(collect_required_column_groups());

    // Initialize lazy readers if needed
    if (column_group_lazy_readers_.empty()) {
      column_group_lazy_readers_.resize(needed_column_groups_.size());
      for (size_t i = 0; i < needed_column_groups_.size(); i++) {
        if (!needed_column_groups_[i]) {
          return arrow::Status::Invalid("Column group at index " + std::to_string(i) + " is null");
        }
        ARROW_ASSIGN_OR_RAISE(column_group_lazy_readers_[i],
                              ColumnGroupLazyReader::create(schema_, needed_column_groups_[i], properties_,
                                                            needed_columns_, key_retriever_callback_));
      }
    }

    // Get tables from each column group
    std::vector<std::shared_ptr<arrow::Table>> tables;
    tables.reserve(column_group_lazy_readers_.size());
    for (const auto& reader : column_group_lazy_readers_) {
      ARROW_ASSIGN_OR_RAISE(auto table, reader->take(row_indices, parallelism));
      tables.emplace_back(std::move(table));
    }

    // Return single table directly
    if (tables.size() == 1) {
      return tables[0];
    }

    // Build column map and validate row counts
    std::unordered_map<std::string, std::pair<size_t, int>> column_map;
    int64_t num_rows = -1;
    for (size_t i = 0; i < tables.size(); ++i) {
      const auto& table = tables[i];
      assert(table);
      if (num_rows == -1) {
        num_rows = table->num_rows();
      } else if (table->num_rows() != num_rows) {
        return arrow::Status::Invalid("Different row counts in column groups");
      }

      const auto& schema = table->schema();
      for (int j = 0; j < table->num_columns(); ++j) {
        column_map[schema->field(j)->name()] = {i, j};
      }
    }

    // Reorder columns to match needed_columns_ order
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    columns.reserve(needed_columns_.size());
    fields.reserve(needed_columns_.size());

    for (const auto& col_name : needed_columns_) {
      auto it = column_map.find(col_name);
      if (it != column_map.end()) {
        const auto& [table_idx, col_idx] = it->second;
        const auto& table = tables[table_idx];
        fields.emplace_back(table->schema()->field(col_idx));
        columns.emplace_back(table->column(col_idx));
      } else {
        // Fill missing column with nulls
        auto field = schema_->GetFieldByName(col_name);
        if (!field) {
          return arrow::Status::Invalid("Column " + col_name + " not found in schema");
        }
        ARROW_ASSIGN_OR_RAISE(auto null_array, arrow::MakeArrayOfNull(field->type(), num_rows));
        fields.emplace_back(field);
        columns.emplace_back(std::make_shared<arrow::ChunkedArray>(null_array));
      }
    }

    return arrow::Table::Make(arrow::schema(fields), columns, num_rows);
  }

  private:
  std::shared_ptr<ColumnGroups> cgs_;        ///< Dataset column groups with metadata and layout info
  std::shared_ptr<arrow::Schema> schema_;    ///< Logical Arrow schema defining data structure
  Properties properties_;                    ///< Configuration properties including encryption
  std::vector<std::string> needed_columns_;  ///< Subset of columns to read (empty = all columns)
  mutable std::vector<std::shared_ptr<ColumnGroup>>
      needed_column_groups_;  ///< Column groups required for needed columns (cached)
  std::function<std::string(const std::string&)>
      key_retriever_callback_;  ///< Callback function for retrieving encryption keys

  std::vector<std::unique_ptr<ColumnGroupLazyReader>> column_group_lazy_readers_;

  /**
   * @brief Collects unique column groups for the requested columns
   */
  arrow::Status collect_required_column_groups() const {
    if (!needed_column_groups_.empty()) {
      return arrow::Status::OK();  // Already initialized
    }

    std::set<std::shared_ptr<ColumnGroup>> unique_groups;

    for (const auto& column_name : needed_columns_) {
      auto column_group = std::find_if(cgs_->begin(), cgs_->end(), [&column_name](const auto& cg) {
        return std::find(cg->columns.begin(), cg->columns.end(), column_name) != cg->columns.end();
      });
      if (column_group == cgs_->end()) {
        continue;  // Skip missing column groups
      }
      unique_groups.insert(*column_group);
    }

    // FIXME: we should support it
    if (unique_groups.empty()) {
      return arrow::Status::Invalid("No column groups found for needed columns");
    }

    needed_column_groups_.assign(unique_groups.begin(), unique_groups.end());
    return arrow::Status::OK();
  }

  void set_keyretriever(const std::function<std::string(const std::string&)>& callback) override {
    key_retriever_callback_ = callback;
  }
};

// ==================== Factory Function Implementation ====================

std::unique_ptr<Reader> Reader::create(const std::shared_ptr<ColumnGroups>& cgs,
                                       const std::shared_ptr<arrow::Schema>& schema,
                                       const std::shared_ptr<std::vector<std::string>>& needed_columns,
                                       const Properties& properties) {
  return std::make_unique<ReaderImpl>(cgs, schema, needed_columns, properties);
}

}  // namespace milvus_storage::api
