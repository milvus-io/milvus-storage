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

#include "common/macro.h"
#include "format/parquet/file_writer.h"
#include <parquet/properties.h>
#include <memory>
#include <string>
#include "filesystem/fs.h"
#include <boost/variant.hpp>
#include "packed/utils/config.h"
#include "packed/utils/serde.h"

namespace milvus_storage {

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     arrow::fs::FileSystem& fs,
                                     const std::string& file_path)
    : schema_(std::move(schema)),
      fs_(fs),
      file_path_(file_path),
      props_(*parquet::default_writer_properties()),
      count_(0) {}

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     arrow::fs::FileSystem& fs,
                                     const std::string& file_path,
                                     const parquet::WriterProperties& props)
    : schema_(std::move(schema)), fs_(fs), file_path_(file_path), props_(props) {}

Status ParquetFileWriter::Init() {
  auto coln = schema_->num_fields();
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto sink, fs_.OpenOutputStream(file_path_));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto writer,
                                parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink));

  writer_ = std::move(writer);
  kv_metadata_ = std::make_shared<arrow::KeyValueMetadata>();
  return Status::OK();
}

Status ParquetFileWriter::Write(const arrow::RecordBatch& record) {
  RETURN_ARROW_NOT_OK(writer_->WriteRecordBatch(record));
  count_ += record.num_rows();
  return Status::OK();
}

Status ParquetFileWriter::WriteTable(const arrow::Table& table) {
  RETURN_ARROW_NOT_OK(writer_->WriteTable(table));
  count_ += table.num_rows();
  return Status::OK();
}

Status ParquetFileWriter::WriteRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                                             const std::vector<size_t>& batch_memory_sizes) {
  auto WriteRowGroup = [&](const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch, size_t group_size) -> Status {
    row_group_sizes_.push_back(group_size);
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto table, arrow::Table::FromRecordBatches(batch));
    RETURN_ARROW_NOT_OK(writer_->WriteTable(*table));
    return Status::OK();
  };

  size_t current_size = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> current_batches;
  for (int i = 0; i < batches.size(); i++) {
    if (current_size + batch_memory_sizes[i] >= DEFAULT_MAX_ROW_GROUP_SIZE && !current_batches.empty()) {
      RETURN_ARROW_NOT_OK(WriteRowGroup(current_batches, current_size));
      current_batches.clear();
      current_size = 0;
    }
    current_batches.push_back(batches[i]);
    current_size += batch_memory_sizes[i];
  }
  if (!current_batches.empty()) {
    RETURN_ARROW_NOT_OK(WriteRowGroup(current_batches, current_size));
  }
  return Status::OK();
}

int64_t ParquetFileWriter::count() { return count_; }

Status ParquetFileWriter::Close() {
  std::string meta = PackedMetaSerde::serialize(row_group_sizes_);
  kv_metadata_->Append(ROW_GROUP_SIZE_META_KEY, meta);
  RETURN_ARROW_NOT_OK(writer_->AddKeyValueMetadata(kv_metadata_));
  RETURN_ARROW_NOT_OK(writer_->Close());
  return Status::OK();
}
}  // namespace milvus_storage
