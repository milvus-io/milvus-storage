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
#include <memory>

namespace milvus_storage::api {

class MemoryBasedChunkSplitStrategy : public ChunkBatchReader::ChunkSplitStrategy {
  public:
  explicit MemoryBasedChunkSplitStrategy(const ChunkReader* reader, int64_t max_block_memory)
      : reader_(reader), max_block_memory_(max_block_memory) {}

  std::vector<ChunkBatchReader::ChunkBlock> split(const std::vector<int64_t>& chunk_indices) override {
    std::vector<ChunkBatchReader::ChunkBlock> blocks;
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

class ParallelDegreeChunkSplitStrategy : public ChunkBatchReader::ChunkSplitStrategy {
  public:
  explicit ParallelDegreeChunkSplitStrategy(uint64_t parallel_degree) : parallel_degree_(parallel_degree) {}

  std::vector<ChunkBatchReader::ChunkBlock> split(const std::vector<int64_t>& chunk_indices) override {
    std::vector<ChunkBatchReader::ChunkBlock> blocks;
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
      std::vector<ChunkBatchReader::ChunkBlock> continuous_blocks;
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

ChunkBatchReader::ChunkBatchReader(std::vector<std::shared_ptr<ChunkReader>> chunk_readers)
    : chunk_readers_(std::move(chunk_readers)) {}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkBatchReader::read_chunks(
    size_t chunk_reader_index,
    const std::vector<int64_t>& chunk_indices,
    int64_t parallelism,
    int64_t memory_limit) const {
  if (chunk_reader_index >= chunk_readers_.size()) {
    return arrow::Status::Invalid("Chunk reader index out of range");
  }

  if (chunk_indices.empty()) {
    return std::vector<std::shared_ptr<arrow::RecordBatch>>();
  }

  const auto* reader = chunk_readers_[chunk_reader_index].get();

  if (parallelism <= 1 || chunk_indices.size() == 1) {
    return read_chunks_sequential(reader, chunk_indices);
  }

  return read_chunks_parallel(reader, chunk_indices, parallelism, memory_limit);
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkBatchReader::read_chunks_sequential(
    const ChunkReader* reader, const std::vector<int64_t>& chunk_indices) const {
  std::vector<std::shared_ptr<arrow::RecordBatch>> chunks;
  chunks.reserve(chunk_indices.size());

  for (int64_t chunk_index : chunk_indices) {
    ARROW_ASSIGN_OR_RAISE(auto chunk, reader->get_chunk(chunk_index));
    chunks.push_back(chunk);
  }

  return chunks;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ChunkBatchReader::read_chunks_parallel(
    const ChunkReader* reader,
    const std::vector<int64_t>& chunk_indices,
    int64_t parallelism,
    int64_t memory_limit) const {
  std::vector<std::shared_ptr<arrow::RecordBatch>> results;
  results.resize(chunk_indices.size());

  // Create index mapping for original order restoration
  std::unordered_map<int64_t, size_t> index_to_position;
  for (size_t i = 0; i < chunk_indices.size(); ++i) {
    index_to_position[chunk_indices[i]] = i;
  }

  // Choose strategy based on memory limit and parallelism
  std::unique_ptr<ChunkSplitStrategy> strategy;

  if (memory_limit > 0) {
    strategy = std::make_unique<MemoryBasedChunkSplitStrategy>(reader, memory_limit);
  } else {
    strategy = std::make_unique<ParallelDegreeChunkSplitStrategy>(parallelism);
  }

  // Split chunks into optimized blocks
  auto blocks = strategy->split(chunk_indices);

  // Create futures for parallel block processing
  std::vector<std::future<arrow::Result<std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>>>>> futures;
  futures.reserve(blocks.size());

  // Launch parallel tasks for each block using get_chunk_range for continuous blocks
  for (const auto& block : blocks) {
    futures.emplace_back(std::async(std::launch::async, [reader, block]() {
      std::vector<std::pair<int64_t, std::shared_ptr<arrow::RecordBatch>>> block_results;
      block_results.reserve(block.count);

      // Use get_chunk_range for continuous blocks to optimize I/O
      auto range_result = reader->get_chunk_range(block.start_index, block.count);
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