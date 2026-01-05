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

#include <memory>
#include <vector>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/record_batch.h>

#include "milvus-storage/column_groups.h"

namespace milvus_storage {

class FormatWriter {
  public:
  virtual ~FormatWriter() = default;
  /**
   * @brief Write a record batch to the file
   *
   * @param record Record batch to write
   * @return arrow::Status
   */
  [[nodiscard]] virtual arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) = 0;

  /**
   * @brief Flush the writer
   *        After call this function, the data should no longer exist in memory
   * @return arrow::Status
   */
  [[nodiscard]] virtual arrow::Status Flush() = 0;

  /**
   * @brief Close the writer
   *        After call this function, the writer should be closed and cannot be used again
   * @return arrow::Status
   */
  [[nodiscard]] virtual arrow::Result<api::ColumnGroupFile> Close() = 0;
};  // class FormatWriter

}  // namespace milvus_storage
