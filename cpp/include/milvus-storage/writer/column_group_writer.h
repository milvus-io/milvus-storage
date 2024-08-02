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

#include <arrow/record_batch.h>
#include <parquet/properties.h>
#include <memory>
#include "format/parquet/file_writer.h"
#include "arrow/filesystem/filesystem.h"
#include "common/status.h"
#include "writer/column_group.h"

namespace milvus_storage {

class ColumnGroupWriter {
  public:
  ColumnGroupWriter(GroupId group_id,
                    std::shared_ptr<arrow::Schema> schema,
                    arrow::fs::FileSystem& fs,
                    const std::string& file_path,
                    const std::vector<int> origin_column_indices);

  ColumnGroupWriter(GroupId group_id,
                    std::shared_ptr<arrow::Schema> schema,
                    arrow::fs::FileSystem& fs,
                    const std::string& file_path,
                    const parquet::WriterProperties& props,
                    const std::vector<int> origin_column_indices);

  Status Init();
  Status Write(const std::shared_ptr<arrow::RecordBatch>& record);
  Status Flush();
  Status Close();
  GroupId Group_id() const;

  private:
  bool finished_;
  GroupId group_id_;
  ParquetFileWriter writer_;
  ColumnGroup column_group_;
};

}  // namespace milvus_storage
