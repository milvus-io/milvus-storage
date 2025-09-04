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

#include "milvus-storage/packed/column_group_writer.h"
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <parquet/properties.h>

#include <utility>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/packed/column_group.h"
#include <iostream>

namespace milvus_storage {

ColumnGroupWriter::ColumnGroupWriter(GroupId group_id,
                                     std::shared_ptr<arrow::Schema> schema,
                                     std::shared_ptr<arrow::fs::FileSystem> fs,
                                     const std::string& file_path,
                                     const StorageConfig& storage_config,
                                     const std::vector<int>& origin_column_indices,
                                     const std::shared_ptr<parquet::WriterProperties>& writer_props)
    : finished_(false),
      group_id_(group_id),
      column_group_(group_id, origin_column_indices),
      flushed_batches_(0),
      flushed_count_(0),
      flushed_rows_(0) {
  auto builder = parquet::WriterProperties::Builder(*writer_props);
  if (writer_props->file_encryption_properties()) {
    auto deep_copied_decryption = writer_props->file_encryption_properties()->DeepClone();
    builder.encryption(std::move(deep_copied_decryption));
  }
  if (writer_props->default_column_properties().compression() == parquet::Compression::UNCOMPRESSED) {
    builder.compression(parquet::Compression::ZSTD);
    builder.compression_level(3);
  }
  writer_ =
      std::make_unique<ParquetFileWriter>(std::move(schema), std::move(fs), file_path, storage_config, builder.build());
}

Status ColumnGroupWriter::Init() { return writer_->Init(); }

Status ColumnGroupWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  if (finished_) {
    return Status::WriterError("Writer has been closed");
  }
  column_group_.AddRecordBatch(record);
  return Status::OK();
}

Status ColumnGroupWriter::Flush() {
  if (column_group_.Empty()) {
    return Status::OK();
  }

  flushed_count_++;
  RETURN_NOT_OK(writer_->WriteRecordBatches(column_group_.GetRecordBatches(), column_group_.GetRecordMemoryUsages()));
  flushed_batches_ += column_group_.GetRecordBatchNum();
  flushed_rows_ += column_group_.GetTotalRows();
  RETURN_NOT_OK(column_group_.Clear());

  return Status::OK();
}

Status ColumnGroupWriter::WriteGroupFieldIDList(const GroupFieldIDList& list) {
  writer_->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, list.Serialize());
  return Status::OK();
}

Status ColumnGroupWriter::AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) {
  for (const auto& [key, value] : metadata) {
    writer_->AppendKVMetadata(key, value);
  }
  return Status::OK();
}

Status ColumnGroupWriter::Close() {
  if (finished_) {
    return Status::OK();
  }
  finished_ = true;
  LOG_STORAGE_DEBUG_ << "Group " << group_id_ << " flushed " << flushed_batches_ << " batches and " << flushed_rows_
                     << " rows in " << flushed_count_ << " flushes";
  return writer_->Close();
}

GroupId ColumnGroupWriter::Group_id() const { return group_id_; }

}  // namespace milvus_storage
