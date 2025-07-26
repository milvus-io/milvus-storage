// Copyright 2024 Zilliz
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

#include <arrow/array/array_base.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/properties.h>
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/packed/chunk_manager.h"
#include <set>
#include <queue>
#include <iostream>

namespace milvus_storage {

ChunkManager::ChunkManager(const std::vector<ColumnOffset>& column_offsets, int64_t chunksize)
    : column_offsets_(column_offsets), chunksize_(chunksize) {
  chunk_states_ = std::vector<ChunkState>(column_offsets_.size());
}

std::vector<std::shared_ptr<arrow::ArrayData>> ChunkManager::SliceChunksByMaxContiguousSlice(
    int64_t chunksize, std::vector<std::queue<std::shared_ptr<arrow::Table>>>& tables) {
  // Determine the maximum contiguous slice across all tables)
  SetChunkSize(std::min(chunksize, DEFAULT_READ_BATCH_SIZE));
  std::vector<const arrow::Array*> chunks(column_offsets_.size());
  std::vector<int> chunk_sizes(column_offsets_.size());
  std::set<int> table_to_pop;

  // Identify the chunk for each column and adjust chunk size
  for (int i = 0; i < column_offsets_.size(); ++i) {
    auto offset = column_offsets_[i];
    auto& table_queue = tables[offset.path_index];

    if (table_queue.empty() || !table_queue.front()) {
      throw std::runtime_error("Table is empty for path_index: " + std::to_string(offset.path_index));
    }

    auto table = table_queue.front();

    // Check if col_index is within bounds
    if (offset.col_index >= table->num_columns()) {
      throw std::runtime_error("Column index " + std::to_string(offset.col_index) + " out of bounds for table with " +
                               std::to_string(table->num_columns()) + " columns");
    }

    auto column = table->column(offset.col_index);

    auto chunk = column->chunk(chunk_states_[i].chunk).get();

    // Adjust chunksize if a smaller contiguous chunk is found
    SetChunkSize(std::min(chunksize_, chunk->length() - chunk_states_[i].offset));

    chunks[i] = chunk;
    chunk_sizes[i] = column->num_chunks();
  }

  // Slice chunks and advance chunk index as appropriate
  std::vector<std::shared_ptr<arrow::ArrayData>> batch_data(column_offsets_.size());
  for (int i = 0; i < column_offsets_.size(); ++i) {
    auto& chunk_state = chunk_states_[i];
    const auto& chunk = chunks[i];
    int64_t offset = chunk_state.offset;
    std::shared_ptr<arrow::ArrayData> slice_data;

    if (chunk->length() - offset == chunksize_) {
      // If the entire remaining chunk matches the chunksize, move to the next chunk
      chunk_state.addChunk(1);
      chunk_state.resetOffset();
      slice_data = (offset > 0) ? chunk->Slice(offset, chunksize_)->data() : chunk->data();

      // Mark the table to pop if all chunks are consumed
      if (chunk_state.chunk == chunk_sizes[i]) {
        table_to_pop.insert(column_offsets_[i].path_index);
        chunk_state.reset();
      }
    } else {
      chunk_state.incOffset(chunksize_);
      slice_data = chunk->Slice(offset, chunksize_)->data();
    }
    batch_data[i] = std::move(slice_data);
  }

  // Pop fully consumed tables
  for (auto& table_index : table_to_pop) {
    if (!tables[table_index].empty()) {
      auto& table = tables[table_index].front();
      table.reset();
      tables[table_index].pop();
    }
  }
  return batch_data;
}

// resets the chunk states for columns in a specific file.
void ChunkManager::ResetChunkState(int path_index) {
  for (int j = 0; j < column_offsets_.size(); ++j) {
    if (column_offsets_[j].path_index == path_index) {
      chunk_states_[j].reset();
    }
  }
}

}  // namespace milvus_storage