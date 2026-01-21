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
#ifdef BUILD_LANCE_BRIDGE
#include "milvus-storage/format/lance/lance_table_reader.h"

#include <string>
#include <iostream>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "lance_bridge.h"

namespace milvus_storage::lance {

static const std::string URI_DELIMITER = "?fragment_id=";

arrow::Result<std::pair<std::string, uint64_t>> LanceTableReader::parse_uri(const std::string& uri) {
  // uri format: {base_path}?fragment_id={fragment_id}
  uint64_t fragment_id = 0;
  auto pos = uri.find(URI_DELIMITER);
  if (pos == std::string::npos) {
    return arrow::Status::Invalid("Invalid uri format: ", uri,
                                  ". Expected format: {base_path}?fragment_id={fragment_id}");
  }
  try {
    fragment_id = std::stoull(uri.substr(pos + URI_DELIMITER.length()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Invalid fragment_id in uri: {}", uri));
  }
  auto base_path = uri.substr(0, pos);
  return std::make_pair(base_path, fragment_id);
}

LanceTableReader::LanceTableReader(const std::shared_ptr<BlockingDataset> dataset,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const milvus_storage::api::Properties& properties)
    : dataset_(dataset),
      fragment_id_(fragment_id),
      schema_(schema),
      properties_(properties),
      fragment_reader_(nullptr) {
  assert(schema_);
}

LanceTableReader::LanceTableReader(const std::string& uri,
                                   uint64_t fragment_id,
                                   const std::shared_ptr<arrow::Schema>& schema,
                                   const milvus_storage::api::Properties& properties)
    : uri_(uri), fragment_id_(fragment_id), schema_(schema), properties_(properties), fragment_reader_(nullptr) {
  assert(schema_);
}

static std::vector<RowGroupInfo> create_row_group_infos(uint64_t rows_in_file, uint64_t logical_chunk_rows) {
  if (rows_in_file == 0) {
    return std::vector<RowGroupInfo>();
  }

  std::vector<RowGroupInfo> result;
  uint64_t last_offset = 0;

  while (last_offset < rows_in_file) {
    uint64_t end_offset = std::min(last_offset + logical_chunk_rows, rows_in_file);
    result.emplace_back(RowGroupInfo{
        .start_offset = last_offset,
        .end_offset = end_offset,
        .memory_size = 1,  // TODO: measure the memory size of each chunk in lance
    });
    last_offset = end_offset;
  }

  return result;
}

arrow::Status LanceTableReader::open() {
  assert(!fragment_reader_);
  ArrowSchema c_arrow_schema;

  if (!dataset_) {
    dataset_ = BlockingDataset::Open(uri_);
  }

  ARROW_ASSIGN_OR_RAISE(logical_chunk_rows_, api::GetValue<uint64_t>(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS));
  ARROW_RETURN_NOT_OK(arrow::ExportSchema(*schema_, &c_arrow_schema));

  fragment_reader_ = BlockingFragmentReader::Open(*dataset_, fragment_id_, c_arrow_schema);
  row_group_infos_ = create_row_group_infos(fragment_reader_->RowCount(), logical_chunk_rows_);

  return arrow::Status::OK();
}

arrow::Result<std::vector<RowGroupInfo>> LanceTableReader::get_row_group_infos() {
  assert(fragment_reader_);
  return row_group_infos_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> LanceTableReader::get_chunk(const int& row_group_index) {
  assert(fragment_reader_);
  auto start_idx = row_group_infos_[row_group_index].start_offset;
  auto end_idx = row_group_infos_[row_group_index].end_offset;
  ArrowArrayStream array_stream = fragment_reader_->ReadRangesAsStream(start_idx, end_idx, end_idx - start_idx);
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
  assert(chunkedarray != nullptr && chunkedarray->num_chunks() == 1);

  return arrow::RecordBatch::FromStructArray(chunkedarray->chunk(0));
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> LanceTableReader::get_chunks(
    const std::vector<int>& rg_indices_in_file) {
  assert(fragment_reader_);
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;

#ifndef NDEBUG
  // verify rg_indices_in_file have been sorted
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    assert(rg_indices_in_file[i] >= rg_indices_in_file[i - 1]);
  }
#endif

  std::vector<std::pair<uint64_t, uint64_t>> rg_idx_ranges;

  // calc continuous ranges
  // ex. [1, 2, 3, 5] -> [(1, 3), (5, 5)]
  size_t start_idx = 0;
  for (size_t i = 1; i < rg_indices_in_file.size(); ++i) {
    if (rg_indices_in_file[i] != rg_indices_in_file[i - 1] + 1) {
      rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file[i - 1]);
      start_idx = i;
    }
  }

  if (start_idx < rg_indices_in_file.size()) {
    rg_idx_ranges.emplace_back(rg_indices_in_file[start_idx], rg_indices_in_file.back());
  }

  for (const auto& rg_range : rg_idx_ranges) {
    // load continuous chunks in one read
    const auto& start_rg_info = row_group_infos_[rg_range.first];
    const auto& end_rg_info = row_group_infos_[rg_range.second];

    ArrowArrayStream array_stream = fragment_reader_->ReadRangesAsStream(
        start_rg_info.start_offset, end_rg_info.end_offset, end_rg_info.end_offset - start_rg_info.start_offset);
    ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));
    assert(chunkedarray != nullptr);

    // assign to rbs
    for (size_t j = 0; j < chunkedarray->num_chunks(); ++j) {
      ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(j)));
      rbs.emplace_back(rb);
    }
  }

  return rbs;
}

arrow::Result<std::shared_ptr<arrow::Table>> LanceTableReader::take(const std::vector<int64_t>& row_indices) {
  assert(fragment_reader_);
  ArrowArrayStream array_stream = fragment_reader_->TakeAsStream(row_indices, row_indices.size());
  ARROW_ASSIGN_OR_RAISE(auto chunkedarray, arrow::ImportChunkedArray(&array_stream));

  // out of range
  if (chunkedarray->num_chunks() == 0) {
    return arrow::Status::Invalid(fmt::format("out of row range [0, {}]", fragment_reader_->RowCount()));
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  for (size_t i = 0; i < chunkedarray->num_chunks(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::FromStructArray(chunkedarray->chunk(i)));
    rbs.emplace_back(rb);
  }

  return arrow::Table::FromRecordBatches(rbs);
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> LanceTableReader::read_with_range(const uint64_t& start_offset,
                                                                                           const uint64_t& end_offset) {
  assert(fragment_reader_);
  ArrowArrayStream array_stream =
      fragment_reader_->ReadRangesAsStream(start_offset, end_offset, end_offset - start_offset);
  return arrow::ImportRecordBatchReader(&array_stream);
}

arrow::Result<std::shared_ptr<FormatReader>> LanceTableReader::clone_reader() {
  assert(fragment_reader_);  // already opened
  return this->shared_from_this();
}

}  // namespace milvus_storage::lance
#endif  // BUILD_LANCE_BRIDGE
