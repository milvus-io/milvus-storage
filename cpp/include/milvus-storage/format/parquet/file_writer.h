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

#include <parquet/properties.h>
#include <memory>
#include "arrow/filesystem/filesystem.h"
#include "format/writer.h"
#include "parquet/arrow/writer.h"
#include "arrow/table.h"
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

class ParquetFileWriter : public FileWriter {
  static constexpr int64_t DEFAULT_MAX_ROW_GROUP_SIZE = 1024 * 1024;  // 1 MB

  public:
  // with default WriterProperties
  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema, arrow::fs::FileSystem& fs, const std::string& file_path);

  // with custom WriterProperties
  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                    arrow::fs::FileSystem& fs,
                    const std::string& file_path,
                    const parquet::WriterProperties& props);

  Status Init() override;

  Status Write(const arrow::RecordBatch& record) override;

  Status WriteTable(const arrow::Table& table) override;

  Status WriteRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                            const std::vector<size_t>& batch_memory_sizes);

  int64_t count() override;

  Status Close() override;

  private:
  arrow::fs::FileSystem& fs_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;

  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata_;
  parquet::WriterProperties props_;
  int64_t count_ = 0;
  int row_group_num_ = 0;
};
}  // namespace milvus_storage
