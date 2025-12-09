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

#include <vector>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/record_batch.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

struct RowGroupInfo {
  public:
  size_t start_offset;
  size_t end_offset;
  size_t memory_size;

  RowGroupInfo() = default;
  std::string ToString() const;
};

/**
 * FormatReader is a reader to read the format file.
 * It exists both blocking and streaming read interfaces.
 * Blocking interface:
 *   - get_chunk
 *   - get_chunks
 *   - take
 * Streaming interface:
 *   - read_with_range
 *
 */
class FormatReader {
  public:
  virtual ~FormatReader() = default;

  // open the format reader, usage to initialize the reader
  // `open` is typically used to open the file's footer.
  [[nodiscard]] virtual arrow::Status open() = 0;

  // get the row group infos
  [[nodiscard]] virtual arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() = 0;

  // get the chunk
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) = 0;

  // get the chunks
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) = 0;

  // take the rows
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> take(
      const std::vector<uint64_t>& row_indices) = 0;

  // create a streaming reader to read the rows in the range
  //
  // If current format reader support `get_row_group_infos`,
  // then the response from the streaming reader should no
  // longer be split from the middle of the row group. Otherwise,
  // there will be an additional memory copy.
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) = 0;

  // create format reader
  static arrow::Result<std::unique_ptr<FormatReader>> create(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& format,
      const std::string& path,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever);

};  // class FormatReader

}  // namespace milvus_storage
