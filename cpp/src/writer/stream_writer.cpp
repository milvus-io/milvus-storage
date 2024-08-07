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

#include "writer/stream_writer.h"
#include <cstddef>
#include "common/status.h"
#include "writer/column_group.h"
#include "writer/column_group_writer.h"
#include "writer/splitter/indices_based_splitter.h"
#include "writer/splitter/size_based_splitter.h"

using namespace std;

namespace milvus_storage {

StreamWriter::StreamWriter(size_t memory_limit,
                           std::shared_ptr<arrow::Schema> schema,
                           arrow::fs::FileSystem& fs,
                           std::string& file_path,
                           parquet::WriterProperties& props)
    : memory_limit_(memory_limit),
      schema_(std::move(schema)),
      fs_(fs),
      file_path_(file_path),
      props_(props),
      splitter_({}),
      current_memory_usage_(0) {}

Status StreamWriter::Init(const std::shared_ptr<arrow::RecordBatch>& record) {
  // split first batch into column groups
  std::vector<ColumnGroup> groups = SizeBasedSplitter(record->num_columns()).Split(record);

  // init column group writer and
  // put column groups into max heap
  std::vector<std::vector<int>> group_indices;
  GroupId group_id = 0;
  for (const ColumnGroup& group : groups) {
    std::string group_path = file_path_ + "/" + std::to_string(group_id);
    auto writer =
        std::make_unique<ColumnGroupWriter>(group_id, schema_, fs_, group_path, props_, group.GetOriginColumnIndices());
    auto status = writer->Init();
    if (!status.ok()) {
      return status;
    }
    status = writer->Write(group.GetRecordBatch(0));
    if (!status.ok()) {
      return status;
    }
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(std::move(group));
    group_indices.emplace_back(group.GetOriginColumnIndices());
    group_writers_.emplace_back(std::move(writer));
    group_id++;
  }
  splitter_ = IndicesBasedSplitter(group_indices);
  return Status::OK();
}

Status StreamWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  // keep writing large column groups until memory usage is less than memory limit
  while (current_memory_usage_ > memory_limit_ && !max_heap_.empty()) {
    ColumnGroup max_group = max_heap_.top();
    max_heap_.pop();
    ColumnGroupWriter* writer = group_writers_[max_group.group_id()].get();

    auto status = writer->Flush();
    if (!status.ok()) {
      return status;
    }
    current_memory_usage_ -= max_group.GetMemoryUsage();
  }

  // split record batch into column groups and put them into buffer
  std::vector<ColumnGroup> column_groups = splitter_.Split(record);
  for (const ColumnGroup& group : column_groups) {
    ColumnGroupWriter* writer = group_writers_[group.group_id()].get();
    auto status = writer->Write(group.GetRecordBatch(0));
    if (!status.ok()) {
      return status;
    }
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(std::move(group));
  }
  return Status::OK();
}

Status StreamWriter::Close() {
  for (auto& writer : group_writers_) {
    auto status = writer->Close();
    if (!status.ok()) {
      return status;
    }
  }
  return Status::OK();
}

}  // namespace milvus_storage
