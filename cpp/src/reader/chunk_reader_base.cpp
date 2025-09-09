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
#include <future>
#include <thread>
#include <algorithm>
#include <unordered_map>

namespace milvus_storage::api {

// Helper classes for chunk splitting strategies
class MemoryBasedChunkSplitStrategy {
  public:
  struct ChunkBlock {
    int64_t start_index;
    int64_t count;
  };

  explicit MemoryBasedChunkSplitStrategy(const ChunkReader* reader, int64_t max_block_memory)
      : reader_(reader), max_block_memory_(max_block_memory) {}

  std::vector<ChunkBlock> split(const std::vector<int64_t>& chunk_indices) {
    std::vector<ChunkBlock> blocks;
    if (chunk_indices.empty()) {
      return blocks;
    }

    std::vector<int64_t> sorted_chunk_indices = chunk_indices;
    std::sort(sorted_chunk_indices.begin(), sorted_chunk_indices.end());

    int64_t current_start = sorted_chunk_indices[0];
    int64_t current_count = 1;
    int64_t current_memory = reader_->get_chunk_size(current_start).ValueOr(0);

    for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
      int64_t next_chunk = sorted_chunk_indices[i];
      int64_t next_memory = reader_->get_chunk_size(next_chunk).ValueOr(0);

      if (next_chunk == current_start + current_count && current_memory + next_memory <= max_block_memory_) {
        current_count++;
        current_memory += next_memory;
        continue;
      }

      blocks.push_back({current_start, current_count});
      current_start = next_chunk;
      current_count = 1;
      current_memory = next_memory;
    }

    if (current_count > 0) {
      blocks.push_back({current_start, current_count});
    }

    return blocks;
  }

  private:
  const ChunkReader* reader_;
  int64_t max_block_memory_;
};

class ParallelDegreeChunkSplitStrategy {
  public:
  struct ChunkBlock {
    int64_t start_index;
    int64_t count;
  };

  explicit ParallelDegreeChunkSplitStrategy(uint64_t parallel_degree) : parallel_degree_(parallel_degree) {}

  std::vector<ChunkBlock> split(const std::vector<int64_t>& chunk_indices) {
    std::vector<ChunkBlock> blocks;
    if (chunk_indices.empty()) {
      return blocks;
    }

    std::vector<int64_t> sorted_chunk_indices = chunk_indices;
    std::sort(sorted_chunk_indices.begin(), sorted_chunk_indices.end());

    uint64_t actual_parallel_degree = std::min(parallel_degree_, static_cast<uint64_t>(sorted_chunk_indices.size()));
    if (actual_parallel_degree == 0) {
      return blocks;
    }

    auto create_continuous_blocks = [&](size_t max_block_size = SIZE_MAX) {
      std::vector<ChunkBlock> continuous_blocks;
      int64_t current_start = sorted_chunk_indices[0];
      int64_t current_count = 1;

      for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
        int64_t next_chunk = sorted_chunk_indices[i];

        if (next_chunk == current_start + current_count && current_count < max_block_size) {
          current_count++;
          continue;
        }
        continuous_blocks.push_back({current_start, current_count});
        current_start = next_chunk;
        current_count = 1;
      }

      if (current_count > 0) {
        continuous_blocks.push_back({current_start, current_count});
      }
      return continuous_blocks;
    };

    if (sorted_chunk_indices.size() <= actual_parallel_degree) {
      return create_continuous_blocks();
    }

    size_t avg_block_size = (sorted_chunk_indices.size() + actual_parallel_degree - 1) / actual_parallel_degree;

    return create_continuous_blocks(avg_block_size);
  }

  private:
  uint64_t parallel_degree_;
};

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunk_range(
    int64_t start_chunk_index, int64_t chunk_count) const {
  // Default implementation: read chunks sequentially using get_chunk
  std::vector<std::shared_ptr<arrow::RecordBatch>> chunks;
  chunks.reserve(chunk_count);

  for (int64_t i = 0; i < chunk_count; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto chunk, get_chunk(start_chunk_index + i));
    chunks.push_back(chunk);
  }

  return chunks;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunks(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism, int64_t memory_limit) const {
  if (chunk_indices.empty()) {
    return std::vector<std::shared_ptr<arrow::RecordBatch>>();
  }

  // Single chunk case - use direct get_chunk
  if (chunk_indices.size() == 1) {
    ARROW_ASSIGN_OR_RAISE(auto chunk, get_chunk(chunk_indices[0]));
    return std::vector<std::shared_ptr<arrow::RecordBatch>>{chunk};
  }

  // Sequential execution for simple cases
  if (parallelism <= 1) {
    return get_chunks_sequential(chunk_indices);
  }

  // Parallel execution with strategy
  return get_chunks_parallel(chunk_indices, parallelism, memory_limit);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunks_sequential(
    const std::vector<int64_t>& chunk_indices) const {
  std::vector<std::shared_ptr<arrow::RecordBatch>> results;
  results.reserve(chunk_indices.size());

  // Try to use continuous chunks optimization
  std::vector<int64_t> sorted_indices = chunk_indices;
  std::sort(sorted_indices.begin(), sorted_indices.end());

  size_t i = 0;
  while (i < sorted_indices.size()) {
    int64_t start_idx = sorted_indices[i];
    int64_t count = 1;

    // Find continuous range
    while (i + count < sorted_indices.size() && sorted_indices[i + count] == start_idx + count) {
      count++;
    }

    if (count > 1) {
      // Use get_chunk_range for continuous chunks
      ARROW_ASSIGN_OR_RAISE(auto range_chunks, get_chunk_range(start_idx, count));
      for (auto& chunk : range_chunks) {
        results.push_back(chunk);
      }
    } else {
      // Single chunk
      ARROW_ASSIGN_OR_RAISE(auto chunk, get_chunk(start_idx));
      results.push_back(chunk);
    }

    i += count;
  }

  // Restore original order
  if (!std::is_sorted(chunk_indices.begin(), chunk_indices.end())) {
    std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_map;
    for (size_t j = 0; j < sorted_indices.size(); ++j) {
      chunk_map[sorted_indices[j]] = results[j];
    }

    for (size_t j = 0; j < chunk_indices.size(); ++j) {
      results[j] = chunk_map[chunk_indices[j]];
    }
  }

  return results;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkReader::get_chunks_parallel(
    const std::vector<int64_t>& chunk_indices, int64_t parallelism, int64_t memory_limit) const {
  std::vector<std::shared_ptr<arrow::RecordBatch>> results;
  results.resize(chunk_indices.size());

  // Create index mapping for original order restoration
  std::unordered_map<int64_t, size_t> index_to_position;
  for (size_t i = 0; i < chunk_indices.size(); ++i) {
    index_to_position[chunk_indices[i]] = i;
  }

  // Choose strategy based on memory limit and parallelism
  std::vector<MemoryBasedChunkSplitStrategy::ChunkBlock> blocks;

  if (memory_limit > 0) {
    MemoryBasedChunkSplitStrategy strategy(this, memory_limit);
    blocks = strategy.split(chunk_indices);
  } else {
    ParallelDegreeChunkSplitStrategy strategy(parallelism);
    auto parallel_blocks = strategy.split(chunk_indices);
    // Convert to compatible format
    for (const auto& block : parallel_blocks) {
      blocks.push_back({block.start_index, block.count});
    }
  }

  // Create futures for parallel block processing
  std::vector<std::future<arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>>> futures;
  futures.reserve(blocks.size());

  // Launch parallel tasks for each block
  for (const auto& block : blocks) {
    futures.emplace_back(std::async(std::launch::async, [this, block]() {
      std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>> block_results;
      block_results.reserve(block.count);

      // Use get_chunk_range for continuous blocks to optimize I/O
      auto range_result = this->get_chunk_range(block.start_index, block.count);
      if (!range_result.ok()) {
        return arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>(
            range_result.status());
      }

      auto batches = range_result.ValueOrDie();
      for (int64_t i = 0; i < block.count; ++i) {
        int64_t chunk_idx = block.start_index + i;
        block_results.emplace_back(chunk_idx, batches[i]);
      }

      return arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>(
          std::move(block_results));
    }));
  }

  // Collect results from all blocks and restore original order
  for (auto& future : futures) {
    auto block_result = future.get();
    if (!block_result.ok()) {
      return block_result.status();
    }

    for (const auto& [chunk_idx, batch] : block_result.ValueOrDie()) {
      auto it = index_to_position.find(chunk_idx);
      if (it != index_to_position.end()) {
        results[it->second] = batch;
      }
    }
  }

  return results;
}

}  // namespace milvus_storage::api