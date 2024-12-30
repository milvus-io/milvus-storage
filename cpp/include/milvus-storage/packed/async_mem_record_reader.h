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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include "common/config.h"
#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include "packed/mem_record_reader.h"

namespace milvus_storage {

/**
 * @brief AsyncMemRecordBatchReader class encapsulates logic for reading row groups in parallel
 */
class AsyncMemRecordBatchReader {
  public:
  /**
   * @brief Constructor for AsyncMemRecordBatchReader.
   *
   * @param fs The Arrow file system instance.
   * @param path The file path to be read.
   * @param schema The schema of the data.
   * @param total_buffer_size The total buffer size for reading.
   */
  AsyncMemRecordBatchReader(arrow::fs::FileSystem& fs,
                            const std::string& path,
                            const std::shared_ptr<arrow::Schema>& schema,
                            int64_t total_buffer_size);

  /**
   * @brief Executes the parallel reading of row groups.
   *
   * @return arrow::Status indicating success or failure of the operation.
   */
  arrow::Status Execute();

  /**
   * @brief Access the readers after execution.
   *
   * @return A reference to the vector of AsyncMemRecordBatchReader instances.
   */
  const std::vector<std::shared_ptr<MemRecordBatchReader>> Readers();

  /**
   * @brief Access the results after execution.
   *
   * @return A vecc to the vector of RecordBatch instances.
   */
  const std::shared_ptr<arrow::Table> Table();

  private:
  arrow::fs::FileSystem& fs_;
  std::string path_;
  std::shared_ptr<arrow::Schema> schema_;
  size_t total_row_groups_;
  int64_t total_buffer_size_;
  int64_t cpu_thread_pool_size_;
  std::vector<std::shared_ptr<MemRecordBatchReader>> readers_;
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>> results_;

  /**
   * @brief Splits row groups into multiple batches.
   *
   * @param batch_size Number of row groups per batch.
   * @return A vector of batches, where each batch is a pair (row_group_offset, num_row_groups).
   */
  std::vector<std::pair<size_t, size_t>> CreateBatches(size_t batch_size) const;
};

}  // namespace milvus_storage