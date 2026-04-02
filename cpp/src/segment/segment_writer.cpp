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

#include "milvus-storage/segment/segment_writer.h"

#include <arrow/array/builder_binary.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include <unordered_map>
#include <unordered_set>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/lob_column/lob_column_reader.h"
#include "milvus-storage/lob_column/lob_column_writer.h"
#include "milvus-storage/lob_column/lob_reference.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::segment {

// implementation of SegmentWriter using api::Writer for regular columns
class SegmentWriterImpl : public SegmentWriter {
  public:
  SegmentWriterImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::shared_ptr<arrow::Schema> original_schema,
                    std::shared_ptr<arrow::Schema> storage_schema,
                    const SegmentWriterConfig& config,
                    std::vector<int> lob_column_indices)
      : fs_(std::move(fs)),
        original_schema_(std::move(original_schema)),
        storage_schema_(std::move(storage_schema)),
        config_(config),
        lob_column_indices_(std::move(lob_column_indices)),
        closed_(false),
        written_rows_(0) {}

  ~SegmentWriterImpl() override {
    if (!closed_) {
      // best effort abort, ignore errors in destructor
      (void)Abort();
    }
  }

  arrow::Status Init() {
    // create LobColumnWriters for each LOB column (TEXT or BINARY)
    for (int col_idx : lob_column_indices_) {
      auto field = original_schema_->field(col_idx);
      auto field_id = GetFieldId(field);
      if (field_id < 0) {
        return arrow::Status::Invalid("LOB column must have a valid field_id in metadata");
      }

      auto it = config_.lob_columns.find(field_id);
      if (it == config_.lob_columns.end()) {
        return arrow::Status::Invalid("LOB column config not found for field_id: " + std::to_string(field_id));
      }

      ARROW_ASSIGN_OR_RAISE(auto writer, lob_column::CreateLobColumnWriter(fs_, it->second));
      lob_writers_[col_idx] = std::move(writer);

      // if rewrite_mode, also create a reader to decode old LOB references
      if (it->second.rewrite_mode) {
        ARROW_ASSIGN_OR_RAISE(auto reader, lob_column::CreateLobColumnReader(fs_, it->second));
        lob_readers_[col_idx] = std::move(reader);
        rewrite_columns_.insert(col_idx);
      }
    }

    // create api::Writer for regular columns using ColumnGroupPolicy
    ARROW_ASSIGN_OR_RAISE(auto policy,
                          api::ColumnGroupPolicy::create_column_group_policy(config_.properties, storage_schema_));

    writer_ = api::Writer::create(config_.segment_path, storage_schema_, std::move(policy), config_.properties);

    return arrow::Status::OK();
  }

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch>& batch) override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    if (!batch || batch->num_rows() == 0) {
      return arrow::Status::OK();
    }

    // validate schema — in rewrite mode, TEXT columns may arrive as binary instead of utf8
    if (!rewrite_columns_.empty()) {
      // relaxed check: verify field count matches
      if (batch->num_columns() != original_schema_->num_fields()) {
        return arrow::Status::Invalid("batch column count does not match writer schema");
      }
    } else {
      if (!batch->schema()->Equals(original_schema_)) {
        return arrow::Status::Invalid("batch schema does not match writer schema");
      }
    }

    // process each column
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(batch->num_columns());

    for (int i = 0; i < batch->num_columns(); i++) {
      auto it = lob_writers_.find(i);
      if (it != lob_writers_.end()) {
        if (rewrite_columns_.count(i) > 0) {
          // REWRITE mode: input is BinaryArray of LOB references
          // decode references → get raw text → re-encode via writer
          auto ref_array = std::static_pointer_cast<arrow::BinaryArray>(batch->column(i));
          auto reader_it = lob_readers_.find(i);

          // decode all references in batch via reader
          ARROW_ASSIGN_OR_RAISE(auto decoded_array, reader_it->second->ReadArrowArray(ref_array));

          // re-encode decoded data via writer (WriteArrowArray accepts BinaryArray)
          ARROW_ASSIGN_OR_RAISE(auto new_ref_array, it->second->WriteArrowArray(decoded_array));
          columns.push_back(new_ref_array);
        } else {
          // Normal mode: input is raw data (utf8 for TEXT, binary for BINARY LOB)
          auto data_array = std::static_pointer_cast<arrow::BinaryArray>(batch->column(i));
          ARROW_ASSIGN_OR_RAISE(auto ref_array, it->second->WriteArrowArray(data_array));
          columns.push_back(ref_array);
        }
      } else {
        // regular column: use directly
        columns.push_back(batch->column(i));
      }
    }

    // create storage batch with LOBReferences
    auto storage_batch = arrow::RecordBatch::Make(storage_schema_, batch->num_rows(), columns);

    // write to api::Writer
    ARROW_RETURN_NOT_OK(writer_->write(storage_batch));

    written_rows_ += batch->num_rows();
    stats_.total_rows = written_rows_;

    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    if (closed_) {
      return arrow::Status::Invalid("writer is closed");
    }

    // flush all TEXT column writers
    for (auto& [col_idx, writer] : lob_writers_) {
      ARROW_RETURN_NOT_OK(writer->Flush());
    }

    // flush api::Writer
    ARROW_RETURN_NOT_OK(writer_->flush());

    return arrow::Status::OK();
  }

  arrow::Result<SegmentWriteOutput> Close() override {
    if (closed_) {
      return arrow::Status::Invalid("writer is already closed");
    }

    // flush pending data
    ARROW_RETURN_NOT_OK(Flush());

    // close all TEXT column writers and collect LOB file results
    for (auto& [col_idx, writer] : lob_writers_) {
      ARROW_ASSIGN_OR_RAISE(auto lob_files, writer->Close());
      lob_file_results_[col_idx] = std::move(lob_files);

      auto text_stats = writer->GetStats();
      stats_.lob_files_created += text_stats.lob_files_created;
    }

    // close LOB readers (rewrite mode)
    for (auto& [col_idx, reader] : lob_readers_) {
      ARROW_RETURN_NOT_OK(reader->Close());
    }

    // close api::Writer and get ColumnGroups
    ARROW_ASSIGN_OR_RAISE(auto column_groups, writer_->close());
    stats_.parquet_files_created = column_groups->size();

    // build LOB file info list from results
    std::vector<api::LobFileInfo> lob_file_infos;
    for (const auto& [col_idx, lob_files] : lob_file_results_) {
      auto field = original_schema_->field(col_idx);
      auto field_id = GetFieldId(field);

      for (const auto& lob_result : lob_files) {
        api::LobFileInfo lob_info;
        lob_info.path = lob_result.path;
        lob_info.field_id = field_id;
        lob_info.total_rows = lob_result.total_rows;
        lob_info.valid_rows = lob_result.valid_rows;
        lob_info.file_size_bytes = lob_result.file_size_bytes;
        lob_file_infos.push_back(std::move(lob_info));
      }
    }

    closed_ = true;

    SegmentWriteOutput output;
    output.column_groups = column_groups;
    output.lob_files = std::move(lob_file_infos);
    output.rows_written = written_rows_;

    return output;
  }

  arrow::Status Abort() override {
    if (closed_) {
      return arrow::Status::OK();
    }

    // abort all TEXT column writers (they will delete their LOB files)
    for (auto& [col_idx, writer] : lob_writers_) {
      // best effort abort, continue on error
      (void)writer->Abort();
    }

    // close LOB readers (rewrite mode)
    for (auto& [col_idx, reader] : lob_readers_) {
      (void)reader->Close();
    }

    closed_ = true;
    return arrow::Status::OK();
  }

  int64_t WrittenRows() const override { return written_rows_; }

  SegmentWriterStats GetStats() const override { return stats_; }

  std::shared_ptr<arrow::Schema> GetStorageSchema() const override { return storage_schema_; }

  std::shared_ptr<arrow::Schema> GetOriginalSchema() const override { return original_schema_; }

  bool IsClosed() const override { return closed_; }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> original_schema_;
  std::shared_ptr<arrow::Schema> storage_schema_;
  SegmentWriterConfig config_;
  std::vector<int> lob_column_indices_;

  bool closed_;
  int64_t written_rows_;

  // LOB column writers (TEXT or BINARY), keyed by column index
  std::unordered_map<int, std::unique_ptr<lob_column::LobColumnWriter>> lob_writers_;

  // LOB column readers for rewrite mode — used to decode old LOB references
  std::unordered_map<int, std::unique_ptr<lob_column::LobColumnReader>> lob_readers_;

  // column indices that are in rewrite mode
  std::unordered_set<int> rewrite_columns_;

  // api::Writer for regular columns (Parquet/Vortex/Lance based on policy)
  std::unique_ptr<api::Writer> writer_;

  // LOB file results, keyed by column index
  std::unordered_map<int, std::vector<lob_column::LobFileResult>> lob_file_results_;

  // statistics
  SegmentWriterStats stats_;
};

// factory function
arrow::Result<std::unique_ptr<SegmentWriter>> SegmentWriter::Create(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                    const std::shared_ptr<arrow::Schema>& schema,
                                                                    const SegmentWriterConfig& config) {
  if (!fs) {
    return arrow::Status::Invalid("filesystem is null");
  }

  if (!schema) {
    return arrow::Status::Invalid("schema is null");
  }

  if (config.segment_path.empty()) {
    return arrow::Status::Invalid("segment_path is empty");
  }

  // validate required properties for ColumnGroupPolicy
  if (config.properties.find(PROPERTY_WRITER_POLICY) == config.properties.end()) {
    return arrow::Status::Invalid("properties must contain " + std::string(PROPERTY_WRITER_POLICY));
  }

  // identify LOB columns (TEXT or BINARY) and build storage schema
  std::vector<std::shared_ptr<arrow::Field>> storage_fields;
  std::vector<int> lob_column_indices;

  for (int i = 0; i < schema->num_fields(); i++) {
    auto field = schema->field(i);

    if (config.lob_columns.count(GetFieldId(field)) > 0) {
      // LOB column: storage always uses binary for LOBReferences
      // (input may be utf8 for TEXT or binary for BINARY LOB)
      auto storage_field = arrow::field(field->name(), arrow::binary(), field->nullable(), field->metadata()->Copy());
      storage_fields.push_back(storage_field);
      lob_column_indices.push_back(i);
    } else {
      // regular column: keep as-is
      storage_fields.push_back(field);
    }
  }

  auto storage_schema = arrow::schema(storage_fields);

  // create writer implementation
  auto writer =
      std::make_unique<SegmentWriterImpl>(std::move(fs), schema, storage_schema, config, std::move(lob_column_indices));

  ARROW_RETURN_NOT_OK(writer->Init());

  return writer;
}

}  // namespace milvus_storage::segment
