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
#include <exception>
#include <stdexcept>
#include <utility>

#include <arrow/type.h>
#include "milvus-storage/common/log.h"
#include <arrow/status.h>
#include <fmt/format.h>

#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/extend_status.h"
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
  auto writer = std::shared_ptr<PackedRecordBatchWriter>(new PackedRecordBatchWriter(
      std::move(fs), paths, std::move(schema), storage_config, column_groups, buffer_size, std::move(writer_props)));
  ARROW_RETURN_NOT_OK(writer->init());
  return writer;
}

arrow::Status PackedRecordBatchWriter::init() {
  if (!schema_) {
    return MakeExtendError(ExtendStatusCode::PackedInvalidArgs, "Packed writer null schema provided");
  }

  if (paths_.size() != group_indices_.size()) {
    return MakeExtendError(ExtendStatusCode::PackedInvalidArgs,
                           fmt::format("Mismatch between paths number and column groups number: {} vs {}",
                                       paths_.size(), group_indices_.size()));
  }

  if (!fs_) {
    return MakeExtendError(ExtendStatusCode::PackedInvalidArgs, "Packed writer null file system provided");
  }

  auto field_id_list = FieldIDList::Make(schema_);
  if (!field_id_list.ok()) {
    return MakeExtendError(ExtendStatusCode::PackedInvalidArgs,
                           fmt::format("Failed to get field id from schema: {}. [schema={}]",
                                       field_id_list.status().ToString(), schema_->ToString(true)),
                           field_id_list.status().ToString());
  }

  // Validate column group indices are within bounds
  int num_fields = schema_->num_fields();
  for (const auto& group_indice : group_indices_) {
    for (int col_index : group_indice) {
      if (col_index < 0 || col_index >= num_fields) {
        return MakeExtendError(ExtendStatusCode::PackedInvalidArgs,
                               fmt::format("Column index out of range: {} (schema has {} fields), [schema={}]",
                                           col_index, num_fields, schema_->ToString(true)));
      }
    }
  }

  group_field_id_list_ = GroupFieldIDList::Make(group_indices_, field_id_list.ValueOrDie());

  splitter_ = IndicesBasedSplitter(group_indices_);
  for (size_t i = 0; i < paths_.size(); ++i) {
    auto column_group_schema = getColumnGroupSchema(schema_, group_indices_[i]);
    auto writer_result = milvus_storage::parquet::ParquetFileWriter::Make(column_group_schema, fs_, paths_[i],
                                                                          storage_config_, writer_props_);
    if (!writer_result.ok()) {
      return WrapExtendError(ExtendStatusCode::PackedStorageIO,
                             fmt::format("Failed to create packed column group writer. [path={}]", paths_[i]),
                             writer_result.status());
    }
    auto writer = std::move(writer_result).ValueOrDie();
    group_writers_.emplace_back(std::move(writer));
  }
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  try {
    // Fault injection point for testing
    FIU_RETURN_ON(FIUKEY_WRITER_WRITE_FAIL,
                  MakeExtendError(ExtendStatusCode::PackedStorageIO,
                                  fmt::format("Injected fault: {}", FIUKEY_WRITER_WRITE_FAIL)));

    if (!record) {
      return arrow::Status::OK();
    }

    for (const auto& group_indice : group_indices_) {
      for (int col_index : group_indice) {
        if (col_index < 0 || col_index >= record->num_columns()) {
          return MakeExtendError(ExtendStatusCode::PackedInvalidArgs,
                                 fmt::format("Record batch column index out of range: {} (record batch has {} columns)",
                                             col_index, record->num_columns()));
        }
      }
    }

    size_t next_batch_size = GetRecordBatchMemorySize(record);

    ARROW_ASSIGN_OR_RAISE(std::vector<ColumnGroup> column_groups, splitter_.Split(record));

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
      auto flush_status = writer->Flush();
      if (!flush_status.ok()) {
        return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to flush packed column group writer",
                               flush_status);
      }
    }

    // After flushing, add the new column groups if memory usage allows
    for (const ColumnGroup& group : column_groups) {
      current_memory_usage_ += group.GetMemoryUsage();
      max_heap_.emplace(group.GrpId(), group.GetMemoryUsage());

      assert(group.GrpId() < group_writers_.size());
      auto& grp_writer = group_writers_[group.GrpId()];
      auto write_status = grp_writer->Write(group.GetRecordBatch(0));
      if (!write_status.ok()) {
        return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to write packed column group", write_status);
      }
    }

    ARROW_RETURN_NOT_OK(balanceMaxHeap());
    return arrow::Status::OK();
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           fmt::format("Packed writer write failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           "Packed writer write with unknown exception failed unexpectedly");
  }
}

arrow::Status PackedRecordBatchWriter::Close() {
  try {
    // Fault injection point for testing
    FIU_RETURN_ON(FIUKEY_WRITER_CLOSE_FAIL,
                  MakeExtendError(ExtendStatusCode::PackedStorageIO,
                                  fmt::format("Injected fault: {}", FIUKEY_WRITER_CLOSE_FAIL)));

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
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           fmt::format("Packed writer close failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           "Packed writer close with unknown exception failed unexpectedly");
  }
}

arrow::Result<std::vector<size_t>> PackedRecordBatchWriter::Tell() const {
  try {
    std::vector<size_t> positions(group_writers_.size());
    for (size_t writer_idx = 0; writer_idx < group_writers_.size(); ++writer_idx) {
      auto tell_result = group_writers_[writer_idx]->Tell();
      if (!tell_result.ok()) {
        return WrapExtendError(ExtendStatusCode::PackedStorageIO,
                               fmt::format("Failed to tell packed column group writer. [writer_index={}]", writer_idx),
                               tell_result.status());
      }
      positions[writer_idx] = tell_result.ValueOrDie();
    }
    return positions;
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           fmt::format("Packed writer tell failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::PackedUnexpected,
                           "Packed writer tell with unknown exception failed unexpectedly");
  }
}

arrow::Status PackedRecordBatchWriter::AddUserMetadata(const std::string& key, const std::string& value) {
  user_metadata_.emplace_back(key, value);
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::flushRemainingBuffer() {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_WRITER_FLUSH_FAIL, MakeExtendError(ExtendStatusCode::PackedStorageIO,
                                                          fmt::format("Injected fault: {}", FIUKEY_WRITER_FLUSH_FAIL)));

  if (closed_) {
    return arrow::Status::OK();
  }

  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    max_heap_.pop();
    auto& grp_writer = group_writers_[max_group.first];

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    current_memory_usage_ -= max_group.second;
    auto flush_status = grp_writer->Flush();
    if (!flush_status.ok()) {
      return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to flush packed column group writer",
                             flush_status);
    }
  }

  for (auto& grp_writer : group_writers_) {
    auto append_status = grp_writer->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, group_field_id_list_.Serialize());
    if (!append_status.ok()) {
      return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to append packed group field id metadata",
                             append_status);
    }

    auto metadata_status = grp_writer->AddUserMetadata(user_metadata_);
    if (!metadata_status.ok()) {
      return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to add packed user metadata", metadata_status);
    }

    auto close_result = grp_writer->Close();
    if (!close_result.ok()) {
      return WrapExtendError(ExtendStatusCode::PackedStorageIO, "Failed to close packed column group writer",
                             close_result.status());
    }
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
