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
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/writer.h"
#include "parquet/arrow/writer.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include <arrow/util/key_value_metadata.h>
#include "milvus-storage/common/config.h"

namespace milvus_storage {

class ParquetFileWriter : public FileWriter {
  public:
  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    const std::string& file_path,
                    const StorageConfig& storage_config,
                    std::shared_ptr<parquet::WriterProperties> writer_props = parquet::default_writer_properties());

  Status Init() override;

  Status Write(const arrow::RecordBatch& record) override;

  Status WriteTable(const arrow::Table& table) override;

  Status WriteRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                            const std::vector<size_t>& batch_memory_sizes);

  void AppendKVMetadata(const std::string& key, const std::string& value);

  int64_t count() override;

  Status Close() override;

  private:
  Status WriteRowGroup(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch, size_t group_size);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;
  const StorageConfig& storage_config_;

  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata_;
  int64_t count_ = 0;
  RowGroupMetadataVector row_group_metadata_;
  std::shared_ptr<parquet::WriterProperties> writer_props_;

  // Cache for remaining batches that are smaller than DEFAULT_MAX_ROW_GROUP_SIZE
  std::vector<std::shared_ptr<arrow::RecordBatch>> cached_batches_;
  size_t cached_size_ = 0;
  bool closed_ = false;
};
}  // namespace milvus_storage
