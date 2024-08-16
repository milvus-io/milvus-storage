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
#include "packed/chunk_manager.h"

namespace milvus_storage {

ChunkManager::ChunkManager(const std::vector<ColumnOffset>& column_offsets, int64_t chunksize)
    : column_offsets_(column_offsets), chunksize_(chunksize) {
  chunk_states_ = std::vector<ChunkState>(column_offsets_.size());
  for (int i = 0; i < column_offsets_.size(); ++i) {
    chunk_states_[i] = ChunkState(0, 0);
  }
}

// Determine the maximum contiguous slice across all tables
std::vector<const arrow::Array*> ChunkManager::GetMaxContiguousSlice(
    const std::vector<std::shared_ptr<arrow::Table>>& tables) {
  std::vector<const arrow::Array*> chunks(column_offsets_.size());
  for (int i = 0; i < column_offsets_.size(); ++i) {
    auto offset = column_offsets_[i];
    auto table = tables[offset.path_index];
    auto column = table->column(offset.col_index);
    auto chunk = column->chunk(chunk_states_[i].chunk).get();
    int64_t chunk_remaining = chunk->length() - chunk_states_[i].offset;
    if (chunk_remaining < chunksize_) {
      chunksize_ = chunk_remaining;
    }

    chunks[i] = chunk;
  }
  return chunks;
}

// Slice chunks and advance chunk index as appropriate
std::vector<std::shared_ptr<arrow::ArrayData>> ChunkManager::SliceChunks(
    const std::vector<const arrow::Array*>& chunks) {
  std::vector<std::shared_ptr<arrow::ArrayData>> batch_data(column_offsets_.size());
  for (int i = 0; i < column_offsets_.size(); ++i) {
    const arrow::Array* chunk = chunks[i];
    auto& chunk_state = chunk_states_[i];
    int64_t offset = chunk_state.offset;
    std::shared_ptr<arrow::ArrayData> slice_data;
    if (chunk->length() - offset == chunksize_) {
      chunk_state.addChunk(1);
      chunk_state.resetOffset();
      slice_data = (offset > 0) ? chunk->Slice(offset, chunksize_)->data() : chunk->data();
    } else {
      chunk_state.incOffset(chunksize_);
      slice_data = chunk->Slice(offset, chunksize_)->data();
    }
    batch_data[i] = std::move(slice_data);
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