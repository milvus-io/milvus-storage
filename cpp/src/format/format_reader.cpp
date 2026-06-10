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

#include "milvus-storage/format/format_reader.h"

#include <limits>
#include <sstream>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/format.h"

namespace milvus_storage {

arrow::Result<std::vector<uint64_t>> DistributeMemorySizes(uint64_t total_size, const std::vector<uint64_t>& weights) {
  if (weights.empty()) {
    return std::vector<uint64_t>{};
  }

  std::vector<uint64_t> result(weights.size(), 0);
  if (total_size == 0) {
    return result;
  }

  uint64_t total_weight = 0;
  for (auto weight : weights) {
    if (weight > std::numeric_limits<uint64_t>::max() - total_weight) {
      return arrow::Status::Invalid("Column memory size weights exceed the uint64_t range");
    }
    total_weight += weight;
  }
  if (total_weight == 0) {
    return arrow::Status::Invalid("Cannot distribute a non-zero memory size with zero column weights");
  }

  uint64_t allocated = 0;
  for (size_t i = 0; i + 1 < weights.size(); ++i) {
    // weights[i] <= total_weight, so the quotient is at most total_size and is safe to cast back to uint64_t.
    result[i] = static_cast<uint64_t>(static_cast<unsigned __int128>(total_size) * weights[i] / total_weight);
    allocated += result[i];
  }
  result.back() = total_size - allocated;
  return result;
}

std::string RowGroupInfo::ToString() const {
  std::stringstream ss;
  ss << "RowGroupInfo{"
     << "start_offset=" << start_offset << ", end_offset=" << end_offset << ", memory_size=" << memory_size << "}";
  return ss.str();
}

arrow::Result<std::shared_ptr<FormatReader>> FormatReader::create(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::string& format,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto* fmt, Format::get(format));
  return fmt->create_reader(read_schema, file, properties, needed_columns, key_retriever);
}

// Default async-shaped methods execute their synchronous counterpart inline;
// native formats override them when they can return before completion.
folly::SemiFuture<arrow::Status> FormatReader::open_async() { return folly::makeSemiFuture(open()); }

folly::SemiFuture<arrow::Result<std::shared_ptr<FormatReader>>> FormatReader::create_async(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::string& format,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  // Preserve the format-specific async factory semantics instead of selecting
  // an executor or wrapping the synchronous factory at this layer.
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto* fmt, Format::get(format));
  return fmt->create_reader_async(read_schema, file, properties, needed_columns, key_retriever);
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>> FormatReader::read_with_range_async(
    uint64_t start_offset, uint64_t end_offset) {
  // Compatibility fallback: read_with_range() may block before this future exists.
  return folly::makeSemiFuture(read_with_range(start_offset, end_offset));
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> FormatReader::take_async(
    const std::vector<int64_t>& row_indices) {
  // Compatibility fallback: take() may block before this future exists.
  return folly::makeSemiFuture(take(row_indices));
}

}  // namespace milvus_storage
