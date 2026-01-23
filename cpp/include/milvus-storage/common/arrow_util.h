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
#include <cstdint>
#include <string>

#include <arrow/table.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/reader.h>
#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/common/constants.h"

namespace milvus_storage {

// extract field_id from Arrow field metadata (returns -1 if not present)
inline int64_t GetFieldId(const std::shared_ptr<arrow::Field>& field) {
  auto metadata = field->metadata();
  if (!metadata || !metadata->Contains(ARROW_FIELD_ID_KEY)) {
    return -1;
  }
  return std::stoll(metadata->Get(ARROW_FIELD_ID_KEY).ValueOrDie());
}
arrow::Result<std::unique_ptr<::parquet::arrow::FileReader>> MakeArrowFileReader(arrow::fs::FileSystem& fs,
                                                                                 const std::string& file_path);

arrow::Result<std::unique_ptr<::parquet::arrow::FileReader>> MakeArrowFileReader(
    arrow::fs::FileSystem& fs, const std::string& file_path, const ::parquet::ReaderProperties& read_properties);

size_t GetRecordBatchMemorySize(const std::shared_ptr<arrow::RecordBatch>& record_batch);

size_t GetArrowArrayMemorySize(const std::shared_ptr<arrow::Array>& array);

size_t GetTableMemorySize(const std::shared_ptr<arrow::Table>& table);

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ConvertTableToRecordBatch(const std::shared_ptr<arrow::Table>& table,
                                                                             bool allow_concat = false);
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ConvertTableToRecordBatchs(
    const std::shared_ptr<arrow::Table>& table);

arrow::Result<std::shared_ptr<arrow::Table>> CopySelectedRows(const std::shared_ptr<arrow::Table>& table,
                                                              const std::vector<int64_t>& indices);

arrow::Result<std::string> GetEnvVar(const char* name);

arrow::Result<std::string> GetEnvVar(const std::string& name);

}  // namespace milvus_storage
