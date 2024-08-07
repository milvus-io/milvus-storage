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

#include "writer/column_group_writer.h"
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <parquet/properties.h>
#include "common/status.h"
#include "format/parquet/file_writer.h"
#include "writer/column_group.h"

using namespace std;
namespace milvus_storage {

ColumnGroupWriter::ColumnGroupWriter(GroupId group_id,
                                     std::shared_ptr<arrow::Schema> schema,
                                     arrow::fs::FileSystem& fs,
                                     const std::string& file_path,
                                     const std::vector<int> origin_column_indices)
    : group_id_(group_id), writer_(schema, fs, file_path), column_group_(group_id, origin_column_indices) {}

ColumnGroupWriter::ColumnGroupWriter(GroupId group_id,
                                     std::shared_ptr<arrow::Schema> schema,
                                     arrow::fs::FileSystem& fs,
                                     const std::string& file_path,
                                     const parquet::WriterProperties& props,
                                     const std::vector<int> origin_column_indices)
    : group_id_(group_id), writer_(schema, fs, file_path, props), column_group_(group_id, origin_column_indices) {}

Status ColumnGroupWriter::Init() { return writer_.Init(); }

Status ColumnGroupWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  if (finished_) {
    return Status::WriterError("Writer has been closed");
  }
  column_group_.AddRecordBatch(record);
  return Status::OK();
}

Status ColumnGroupWriter::Flush() {
  auto status = writer_.WriteTable(*column_group_.Table());
  if (!status.ok()) {
    return status;
  }
  column_group_.Clear();
  return Status::OK();
}

Status ColumnGroupWriter::Close() {
  finished_ = true;
  return writer_.Close();
}

GroupId ColumnGroupWriter::Group_id() const { return group_id_; }

}  // namespace milvus_storage
