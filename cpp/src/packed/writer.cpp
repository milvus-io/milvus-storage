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
#include <new>
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
  auto status = writer->init();
  if (!status.ok()) {
    writer->Abort();
    return status;
  }
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
      return writer_result.status();
    }
    auto writer = std::move(writer_result).ValueOrDie();
    group_writers_.emplace_back(std::move(writer));
  }
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::Write(const std::shared_ptr<arrow::RecordBatch>& record) {
  ARROW_RETURN_NOT_OK(writer_status_.Check());
  return writer_status_.Fail(WriteImpl(record));
}

arrow::Status PackedRecordBatchWriter::WriteImpl(const std::shared_ptr<arrow::RecordBatch>& record) {
  try {
    if (closed_) {
      return arrow::Status::Invalid("Cannot write to closed packed writer");
    }

    // Fault injection point for testing
    FIU_RETURN_ON(
        FIUKEY_WRITER_WRITE_FAIL,
        MakeExtendError(ExtendStatusCode::PackedIO, fmt::format("Injected fault: {}", FIUKEY_WRITER_WRITE_FAIL)));

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

    // Stateful file writers cannot continue after any child failure. Flush one
    // group at a time and only advance heap/memory bookkeeping after success.
    // to ensure that memory usage stays strictly below the limit
    while (current_memory_usage_ + next_batch_size >= buffer_size_ && !max_heap_.empty()) {
      LOG_STORAGE_DEBUG_ << "Current memory usage: " << current_memory_usage_ / 1024 / 1024 << " MB, "
                         << ", flushing column group: " << max_heap_.top().first;
      auto max_group = max_heap_.top();

      milvus_storage::parquet::ParquetFileWriter* writer = group_writers_[max_group.first].get();
      auto flush_status = writer->Flush();
      if (!flush_status.ok()) {
        return flush_status;
      }

      max_heap_.pop();
      assert(current_memory_usage_ >= max_group.second);
      current_memory_usage_ -= max_group.second;
    }

    // After flushing, add the new column groups if memory usage allows
    for (const ColumnGroup& group : column_groups) {
      assert(group.GrpId() < group_writers_.size());
      auto& grp_writer = group_writers_[group.GrpId()];
      auto write_status = grp_writer->Write(group.GetRecordBatch(0));
      if (!write_status.ok()) {
        return write_status;
      }

      current_memory_usage_ += group.GetMemoryUsage();
      max_heap_.emplace(group.GrpId(), group.GetMemoryUsage());
    }

    ARROW_RETURN_NOT_OK(balanceMaxHeap());
    return arrow::Status::OK();
  } catch (const std::bad_alloc&) {
    return arrow::Status::OutOfMemory("Packed writer write ran out of memory");
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated,
                           fmt::format("Packed writer write failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated, "Packed writer write failed unexpectedly");
  }
}

void PackedRecordBatchWriter::Abort() noexcept {
  // No writer_status_.Check(): abort is for the state where the writer has
  // already failed.
  if (closed_) {
    // Already finalized, or already aborted: abort after a successful close is
    // a no-op, so a caller can destroy unconditionally.
    return;
  }
  closed_ = true;
  writer_status_.BeginDiscard();
  buffered_batches_.clear();
  // Each group writer owns its own output stream, so each one has to be asked
  // separately; one refusing to release its upload is not a reason to skip the
  // rest.
  for (auto& writer : group_writers_) {
    if (writer != nullptr) {
      writer->Abort();
    }
  }
  group_writers_.clear();
}

arrow::Status PackedRecordBatchWriter::Close() {
  // Abandon on both failure paths; see FormatWriter::Close in format_writer.h.
  if (auto first_failure = writer_status_.Check(); !first_failure.ok()) {
    Abort();
    return first_failure;
  }
  auto status = writer_status_.Fail(CloseImpl());
  if (!status.ok()) {
    Abort();
  }
  return status;
}

arrow::Status PackedRecordBatchWriter::CloseImpl() {
  try {
    // Check if already closed
    if (closed_) {
      return arrow::Status::OK();
    }

    // Fault injection point for testing
    FIU_RETURN_ON(
        FIUKEY_WRITER_CLOSE_FAIL,
        MakeExtendError(ExtendStatusCode::PackedIO, fmt::format("Injected fault: {}", FIUKEY_WRITER_CLOSE_FAIL)));

    // flush all remaining column groups before closing
    auto status = flushRemainingBuffer();
    if (!status.ok()) {
      return status;
    }
    closed_ = true;
    return arrow::Status::OK();
  } catch (const std::bad_alloc&) {
    return arrow::Status::OutOfMemory("Packed writer close ran out of memory");
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated,
                           fmt::format("Packed writer close failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated, "Packed writer close failed unexpectedly");
  }
}

arrow::Result<std::vector<size_t>> PackedRecordBatchWriter::Tell() const {
  ARROW_RETURN_NOT_OK(writer_status_.Check());
  auto result = TellImpl();
  if (!result.ok()) {
    return writer_status_.Fail(result.status());
  }
  return result;
}

arrow::Result<std::vector<size_t>> PackedRecordBatchWriter::TellImpl() const {
  try {
    std::vector<size_t> positions(group_writers_.size());
    for (size_t writer_idx = 0; writer_idx < group_writers_.size(); ++writer_idx) {
      auto tell_result = group_writers_[writer_idx]->Tell();
      if (!tell_result.ok()) {
        return tell_result.status();
      }
      positions[writer_idx] = tell_result.ValueOrDie();
    }
    return positions;
  } catch (const std::bad_alloc&) {
    return arrow::Status::OutOfMemory("Packed writer tell ran out of memory");
  } catch (const std::exception& e) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated,
                           fmt::format("Packed writer tell failed unexpectedly: {}", e.what()));
  } catch (...) {
    return MakeExtendError(ExtendStatusCode::InternalInvariantViolated, "Packed writer tell failed unexpectedly");
  }
}

arrow::Status PackedRecordBatchWriter::AddUserMetadata(const std::string& key, const std::string& value) {
  ARROW_RETURN_NOT_OK(writer_status_.Check());
  return writer_status_.Fail(AddUserMetadataImpl(key, value));
}

arrow::Status PackedRecordBatchWriter::AddUserMetadataImpl(const std::string& key, const std::string& value) {
  if (closed_) {
    return arrow::Status::Invalid("Cannot add metadata to closed packed writer");
  }
  user_metadata_.emplace_back(key, value);
  return arrow::Status::OK();
}

arrow::Status PackedRecordBatchWriter::flushRemainingBuffer() {
  // Fault injection point for testing
  FIU_RETURN_ON(FIUKEY_WRITER_FLUSH_FAIL, MakeExtendError(ExtendStatusCode::PackedIO,
                                                          fmt::format("Injected fault: {}", FIUKEY_WRITER_FLUSH_FAIL)));

  if (closed_) {
    return arrow::Status::OK();
  }

  while (!max_heap_.empty()) {
    auto max_group = max_heap_.top();
    auto& grp_writer = group_writers_[max_group.first];

    LOG_STORAGE_DEBUG_ << "Flushing remaining column group: " << max_group.first;
    auto flush_status = grp_writer->Flush();
    if (!flush_status.ok()) {
      return flush_status;
    }

    max_heap_.pop();
    assert(current_memory_usage_ >= max_group.second);
    current_memory_usage_ -= max_group.second;
  }

  for (size_t writer_idx = 0; writer_idx < group_writers_.size(); ++writer_idx) {
    auto& grp_writer = group_writers_[writer_idx];
    auto append_status = grp_writer->AppendKVMetadata(GROUP_FIELD_ID_LIST_META_KEY, group_field_id_list_.Serialize());
    if (!append_status.ok()) {
      return append_status;
    }

    auto metadata_status = grp_writer->AddUserMetadata(user_metadata_);
    if (!metadata_status.ok()) {
      return metadata_status;
    }

    auto close_result = grp_writer->Close();
    if (!close_result.ok()) {
      return close_result.status();
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
