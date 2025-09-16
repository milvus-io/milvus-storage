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
#include "parquet/arrow/writer.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include <arrow/util/key_value_metadata.h>
#include "milvus-storage/common/config.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/format/format.h"

namespace milvus_storage::parquet {

class ParquetFileWriter : public internal::api::ColumnGroupWriter {
  public:
  ParquetFileWriter(std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::shared_ptr<arrow::Schema> schema,
                    const milvus_storage::api::WriteProperties& properties);

  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    const std::string& file_path,
                    const milvus_storage::StorageConfig& storage_config,
                    std::shared_ptr<::parquet::WriterProperties> writer_props = ::parquet::default_writer_properties());

  ~ParquetFileWriter() override = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Status Close() override;

  arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) override;

  arrow::Status AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata);

  private:
  arrow::Status WriteRowGroup(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch, size_t group_size);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;
  const milvus_storage::StorageConfig& storage_config_;

  std::unique_ptr<::parquet::arrow::FileWriter> writer_;
  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata_;
  int64_t count_ = 0;
  int64_t bytes_written_ = 0;
  int64_t num_chunks_ = 0;
  milvus_storage::RowGroupMetadataVector row_group_metadata_;
  std::shared_ptr<::parquet::WriterProperties> writer_props_;

  // Cache for batches waiting to be written
  std::vector<std::shared_ptr<arrow::RecordBatch>> cached_batches_;
  size_t cached_size_ = 0;
  std::vector<size_t> cached_batch_sizes_;
  bool closed_ = false;
};
}  // namespace milvus_storage::parquet
