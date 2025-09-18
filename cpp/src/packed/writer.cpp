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

#include "milvus-storage/packed/writer.h"
#include <arrow/type.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/arrow_util.h"

namespace milvus_storage {

PackedRecordBatchWriter::PackedRecordBatchWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 std::vector<std::string>& paths,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 StorageConfig& storage_config,
                                                 std::vector<std::vector<int>>& column_groups,
                                                 size_t buffer_size,
                                                 std::shared_ptr<::parquet::WriterProperties> writer_props)
    : buffer_size_(buffer_size), group_indices_(column_groups), splitter_(column_groups), current_memory_usage_(0) {
  if (!schema) {
    LOG_STORAGE_ERROR_ << "Packed writer null schema provided";
    throw std::runtime_error("Packed writer null schema provided");
  }

  if (paths.size() != group_indices_.size()) {
    LOG_STORAGE_ERROR_ << "Mismatch between paths number and column groups number: " << paths.size() << " vs "
                       << group_indices_.size();
    throw std::runtime_error("Mismatch between paths number and column groups number");
  }

  auto field_id_list = FieldIDList::Make(schema);
  if (!field_id_list.ok()) {
    LOG_STORAGE_ERROR_ << "Failed to get field id from schema: " << schema->ToString();
    throw std::runtime_error("Failed to get field id from schema: " + schema->ToString());
  }
  group_field_id_list_ = GroupFieldIDList::Make(column_groups, field_id_list.value());

  splitter_ = IndicesBasedSplitter(group_indices_);
  for (size_t i = 0; i < paths.size(); ++i) {
    auto column_group_schema = getColumnGroupSchema(schema, group_indices_[i]);
    auto writer = std::make_unique<milvus_storage::parquet::ParquetFileWriter>(column_group_schema, fs, paths[i],
                                                                               storage_config, writer_props);
    group_writers_.emplace_back(std::move(writer));
  }
}

Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  if (!record) {
    return Status::OK();
  }

  size_t next_batch_size = GetRecordBatchMemorySize(record);

  std::vector<ColumnGroup> column_groups = splitter_.Split(record);

  // Flush column groups until there's enough room for the new column groups
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + next_batch_size >= buffer_size_ && !max_heap_.empty()) {
    LOG_STORAGE_DEBUG_ << "Current memory usage: " << current_memory_usage_ / 1024 / 1024 << " MB, "
                       << ", flushing column group: " << max_heap_.top().first;
    auto max_group = max_heap_.top();
    max_heap_.pop();

    assert(current_memory_usage_ >= max_group.second);
    current_memory_usage_ -= max_group.second;

    milvus_storage::parquet::ParquetFileWriter* writer = group_writers_[max_group.first].get();
    RETURN_ARROW_NOT_OK(writer->Flush());
  }

  // After flushing, add the new column groups if memory usage allows
  for (const ColumnGroup& group : column_groups) {
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(group.GrpId(), group.GetMemoryUsage());

    assert(group.GrpId() < group_writers_.size());
    auto& grp_writer = group_writers_[group.GrpId()];
    RETURN_ARROW_NOT_OK(grp_writer->Write(group.GetRecordBatch(0)));
  }

  RETURN_NOT_OK(balanceMaxHeap());
  return Status::OK();
}

Status PackedRecordBatchWriter::Close() {
  // Check if already closed
  if (closed_) {
    return Status::OK();
  }

  // flush all remaining column groups before closing
  auto status = flushRemainingBuffer();
  if (status.ok()) {
    closed_ = true;
  }
  return status;
}

Status PackedRecordBatchWriter::AddUserMetadata(const std::string& key, const std::string& value) {
  user_metadata_.emplace_back(key, value);
  return Status::OK();
}

Status PackedRecordBatchWriter::AddMetadataBuilder(
    const std::string& key, const std::function<std::unique_ptr<MetadataBuilder>()>& builder_factory) {
  if (!builder_factory) {
    return Status::InvalidArgument("Metadata builder factory is null");
  }
  for (auto& grp_writer : group_writers_) {
    auto builder = builder_factory();
    if (!builder) {
      return Status::InvalidArgument("Metadata builder factory returned null");
    }
    RETURN_ARROW_NOT_OK(grp_writer->AddMetadataBuilder(key, std::move(builder)));
  }
  return Status::OK();
}

Status PackedRecordBatchWriter::flushRemainingBuffer() {
  if (closed_) {
    return Status::OK();
  }

  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    auto& grp_writer = group_writers_[max_group.first];

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    current_memory_usage_ -= max_group.second;
    RETURN_ARROW_NOT_OK(grp_writer->Flush());
  }

  for (auto& grp_writer : group_writers_) {
    RETURN_ARROW_NOT_OK(grp_writer->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, group_field_id_list_.Serialize()));
    RETURN_ARROW_NOT_OK(grp_writer->AddUserMetadata(user_metadata_));
    RETURN_ARROW_NOT_OK(grp_writer->Close());
  }

  return Status::OK();
}

Status PackedRecordBatchWriter::balanceMaxHeap() {
  std::unordered_map<GroupId, size_t> group_map;
  while (!max_heap_.empty()) {
    auto pair = max_heap_.top();
    max_heap_.pop();
    group_map[pair.first] += pair.second;
  }

  for (const auto& [gid, gsz] : group_map) {
    max_heap_.emplace(gid, gsz);
  }

  return Status::OK();
}

std::shared_ptr<arrow::Schema> PackedRecordBatchWriter::getColumnGroupSchema(
    const std::shared_ptr<arrow::Schema>& schema, const std::vector<int>& column_indices) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(column_indices.size());
  for (int index : column_indices) {
    fields.emplace_back(schema->field(index));
  }
  return arrow::schema(fields);
}

}  // namespace milvus_storage
