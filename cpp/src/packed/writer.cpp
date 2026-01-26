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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <arrow/type.h>
#include <arrow/util/logging.h>
#include <arrow/status.h>
#include <fmt/format.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage {

PackedRecordBatchWriter::PackedRecordBatchWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                 std::vector<std::string>& paths,
                                                 std::shared_ptr<arrow::Schema> schema,
                                                 StorageConfig& storage_config,
                                                 std::vector<std::vector<int>>& column_groups,
                                                 size_t buffer_size,
                                                 std::shared_ptr<::parquet::WriterProperties> writer_props)
    : fs_(std::move(fs)),
      paths_(paths),
      schema_(std::move(schema)),
      storage_config_(storage_config),
      buffer_size_(buffer_size),
      group_indices_(column_groups),
      splitter_(column_groups),
      current_memory_usage_(0),
      writer_props_(std::move(writer_props)) {}

arrow::Result<std::shared_ptr<PackedRecordBatchWriter>> PackedRecordBatchWriter::Make(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    std::vector<std::string>& paths,
    std::shared_ptr<arrow::Schema> schema,
    StorageConfig& storage_config,
    std::vector<std::vector<int>>& column_groups,
    size_t buffer_size,
    std::shared_ptr<::parquet::WriterProperties> writer_props) {
  auto writer = std::shared_ptr<PackedRecordBatchWriter>(
      new PackedRecordBatchWriter(fs, paths, schema, storage_config, column_groups, buffer_size, writer_props));
  ARROW_RETURN_NOT_OK(writer->init());
  return writer;
}

arrow::Status PackedRecordBatchWriter::init() {
  if (!schema_) {
    return arrow::Status::Invalid("Packed writer null schema provided");
  }

  if (paths_.size() != group_indices_.size()) {
    return arrow::Status::Invalid(fmt::format("Mismatch between paths number and column groups number: {} vs {}",
                                              paths_.size(), group_indices_.size()));
  }

  if (!fs_) {
    return arrow::Status::Invalid("Packed writer null file system provided");
  }

  auto field_id_list = FieldIDList::Make(schema_);
  if (!field_id_list.ok()) {
    return arrow::Status::Invalid(fmt::format("Failed to get field id from schema: {}. [schema={}]",
                                              field_id_list.status().ToString(), schema_->ToString(true)));
  }

  // Validate column group indices are within bounds
  int num_fields = schema_->num_fields();
  for (size_t i = 0; i < group_indices_.size(); ++i) {
    for (int col_index : group_indices_[i]) {
      if (col_index < 0 || col_index >= num_fields) {
        return arrow::Status::Invalid(fmt::format("Column index out of range: {} (schema has {} fields), [schema={}]",
                                                  col_index, num_fields, schema_->ToString(true)));
      }
    }
  }

  group_field_id_list_ = GroupFieldIDList::Make(group_indices_, field_id_list.ValueOrDie());

  splitter_ = IndicesBasedSplitter(group_indices_);
  for (size_t i = 0; i < paths_.size(); ++i) {
    auto column_group_schema = getColumnGroupSchema(schema_, group_indices_[i]);
    ARROW_ASSIGN_OR_RAISE(auto writer, milvus_storage::parquet::ParquetFileWriter::Make(
                                           column_group_schema, fs_, paths_[i], storage_config_, writer_props_));
    group_writers_.emplace_back(std::move(writer));
  }
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_WRITER_WRITE_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_WRITER_WRITE_FAIL)));

  if (!record) {
    return arrow::Status::OK();
  }

  size_t next_batch_size = GetRecordBatchMemorySize(record);

  std::vector<ColumnGroup> column_groups = splitter_.Split(record);

  // Flush column groups until there's enough room for the new column groups
  // to ensure that memory usage stays strictly below the limit
  while (current_memory_usage_ + next_batch_size >= buffer_size_ && !max_heap_.empty()) {
    ARROW_LOG(DEBUG) << "Current memory usage: " << current_memory_usage_ / 1024 / 1024 << " MB, "
                     << ", flushing column group: " << max_heap_.top().first;
    auto max_group = max_heap_.top();
    max_heap_.pop();

    assert(current_memory_usage_ >= max_group.second);
    current_memory_usage_ -= max_group.second;

    milvus_storage::parquet::ParquetFileWriter* writer = group_writers_[max_group.first].get();
    ARROW_RETURN_NOT_OK(writer->Flush());
  }

  // After flushing, add the new column groups if memory usage allows
  for (const ColumnGroup& group : column_groups) {
    current_memory_usage_ += group.GetMemoryUsage();
    max_heap_.emplace(group.GrpId(), group.GetMemoryUsage());

    assert(group.GrpId() < group_writers_.size());
    auto& grp_writer = group_writers_[group.GrpId()];
    ARROW_RETURN_NOT_OK(grp_writer->Write(group.GetRecordBatch(0)));
  }

  ARROW_RETURN_NOT_OK(balanceMaxHeap());
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::Close() {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_WRITER_CLOSE_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_WRITER_CLOSE_FAIL)));

  // Check if already closed
  if (closed_) {
    return arrow::Status::OK();
  }

  // flush all remaining column groups before closing
  auto status = flushRemainingBuffer();
  if (status.ok()) {
    closed_ = true;
  }
  return status;
}

arrow::Status PackedRecordBatchWriter::AddUserMetadata(const std::string& key, const std::string& value) {
  user_metadata_.emplace_back(key, value);
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::flushRemainingBuffer() {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_WRITER_FLUSH_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_WRITER_FLUSH_FAIL)));

  if (closed_) {
    return arrow::Status::OK();
  }

  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    auto& grp_writer = group_writers_[max_group.first];

    ARROW_LOG(DEBUG) << "Flushing remaining column group: " << max_group.first;
    current_memory_usage_ -= max_group.second;
    ARROW_RETURN_NOT_OK(grp_writer->Flush());
  }

  for (auto& grp_writer : group_writers_) {
    ARROW_RETURN_NOT_OK(grp_writer->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, group_field_id_list_.Serialize()));
    ARROW_RETURN_NOT_OK(grp_writer->AddUserMetadata(user_metadata_));
    ARROW_RETURN_NOT_OK(grp_writer->Close());
  }

  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::balanceMaxHeap() {
  std::unordered_map<GroupId, size_t> group_map;
  while (!max_heap_.empty()) {
    auto pair = max_heap_.top();
    max_heap_.pop();
    group_map[pair.first] += pair.second;
  }

  for (const auto& [gid, gsz] : group_map) {
    max_heap_.emplace(gid, gsz);
  }

  return arrow::Status::OK();
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
