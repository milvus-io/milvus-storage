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
namespace milvus_storage {

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                                     arrow::fs::FileSystem& fs,
                                     std::string& file_path)
    : schema_(std::move(schema)), fs_(fs), file_path_(file_path) {}

Status ParquetFileWriter::Init() {
  auto coln = schema_->num_fields();
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto sink, fs_.OpenOutputStream(file_path_));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto writer,
                                parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), sink));

  writer_ = std::move(writer);
  return Status::OK();
}

Status ParquetFileWriter::Write(const arrow::RecordBatch& record) {
  RETURN_ARROW_NOT_OK(writer_->WriteRecordBatch(record));
  count_ += record.num_rows();
  return Status::OK();
}

int64_t ParquetFileWriter::count() { return count_; }

Status ParquetFileWriter::Close() {
  RETURN_ARROW_NOT_OK(writer_->Close());
  return Status::OK();
}
}  // namespace milvus_storage
