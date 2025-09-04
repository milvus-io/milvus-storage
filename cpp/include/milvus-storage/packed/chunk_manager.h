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

#pragma once

#include "milvus-storage/packed/column_group.h"
#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>

namespace milvus_storage {

struct ColumnOffset {
  int path_index;
  int col_index;

  ColumnOffset() = default;

  ColumnOffset(int path_index, int col_index) : path_index(path_index), col_index(col_index) {}

  std::string ToString() const {
    return "path_index: " + std::to_string(path_index) + ", col_index: " + std::to_string(col_index);
  }
};

// record which chunk is in use and its offset in the file
struct ChunkState {
  public:
  int chunk;
  int64_t offset;

  ChunkState() : chunk(0), offset(0) {}

  ChunkState(int chunk, int64_t offset) : chunk(chunk), offset(offset) {}

  void reset() {
    resetOffset();
    resetChunk();
  }

  void resetOffset() { this->offset = 0; }

  void incOffset(int64_t delta) { this->offset += delta; }

  void resetChunk() { this->chunk = 0; }

  void addChunk(int chunk) { this->chunk += chunk; }
};

class ChunkManager {
  public:
  ChunkManager(const std::vector<ColumnOffset>& column_offsets, int64_t chunksize);

  std::vector<std::shared_ptr<arrow::ArrayData>> SliceChunksByMaxContiguousSlice(
      int64_t chunksize, std::vector<std::queue<std::shared_ptr<arrow::Table>>>& tables);

  void ResetChunkState(int path_index);

  int64_t GetChunkSize() const { return chunksize_; }

  void SetChunkSize(int64_t chunksize) { chunksize_ = chunksize; }

  private:
  std::vector<ColumnOffset> column_offsets_;
  std::vector<ChunkState> chunk_states_;
  int64_t chunksize_;
};

}  // namespace milvus_storage