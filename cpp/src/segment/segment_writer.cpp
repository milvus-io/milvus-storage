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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/lob_column/lob_column_writer.h"
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
    // open Transaction for manifest management at segment level
    // manifest is stored at: {segment_path}/_metadata/manifest-{version}.avro
    ARROW_ASSIGN_OR_RAISE(transaction_,
                          api::transaction::Transaction::Open(fs_, config_.segment_path, config_.read_version,
                                                              api::transaction::FailResolver, config_.retry_limit));

    // create LobColumnWriters for each TEXT column
    for (int col_idx : lob_column_indices_) {
      auto field = original_schema_->field(col_idx);
      auto field_id = GetFieldId(field);
      if (field_id < 0) {
        return arrow::Status::Invalid("TEXT column must have a valid field_id in metadata");
      }

      auto it = config_.lob_columns.find(field_id);
      if (it == config_.lob_columns.end()) {
        return arrow::Status::Invalid("TEXT column config not found for field_id: " + std::to_string(field_id));
      }

      ARROW_ASSIGN_OR_RAISE(auto writer, lob_column::CreateLobColumnWriter(fs_, it->second));
      lob_writers_[col_idx] = std::move(writer);
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

    // validate schema matches
    if (!batch->schema()->Equals(original_schema_)) {
      return arrow::Status::Invalid("batch schema does not match writer schema");
    }

    // process each column
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(batch->num_columns());

    for (int i = 0; i < batch->num_columns(); i++) {
      auto it = lob_writers_.find(i);
      if (it != lob_writers_.end()) {
        // TEXT column: write to LOB file and get LOBReference
        auto text_array = std::static_pointer_cast<arrow::StringArray>(batch->column(i));
        ARROW_ASSIGN_OR_RAISE(auto ref_array, it->second->WriteArrowArray(text_array));
        columns.push_back(ref_array);
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

  arrow::Result<SegmentWriterResult> Close() override {
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

    // close api::Writer and get ColumnGroups
    ARROW_ASSIGN_OR_RAISE(auto column_groups, writer_->close());
    stats_.parquet_files_created = column_groups->size();

    // add all column groups to transaction
    // NOTE: LOB .vx files are NOT added as column groups - they have their own
    // dedicated lob_files_ section in the manifest (see AddLobFile below)
    transaction_->AppendFiles(*column_groups);

    // add LOB file metadata to transaction
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

        transaction_->AddLobFile(lob_info);
      }
    }

    // commit transaction - this writes the manifest file
    ARROW_ASSIGN_OR_RAISE(auto committed_version, transaction_->Commit());

    closed_ = true;

    // build result
    SegmentWriterResult result;
    result.manifest_path = get_manifest_filepath(config_.segment_path, committed_version);
    result.committed_version = committed_version;
    result.rows_written = written_rows_;

    return result;
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

    // writer_ cleanup is handled by destructor
    // note: api::Writer doesn't have an explicit Abort method
    // Transaction is not committed, so manifest won't be written

    closed_ = true;
    return arrow::Status::OK();
  }

  int64_t WrittenRows() const override { return written_rows_; }

  SegmentWriterStats GetStats() const override { return stats_; }

  std::shared_ptr<arrow::Schema> GetStorageSchema() const override { return storage_schema_; }

  std::shared_ptr<arrow::Schema> GetOriginalSchema() const override { return original_schema_; }

  bool IsClosed() const override { return closed_; }

  int64_t GetReadVersion() const override {
    if (transaction_) {
      return transaction_->GetReadVersion();
    }
    return -1;
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> original_schema_;
  std::shared_ptr<arrow::Schema> storage_schema_;
  SegmentWriterConfig config_;
  std::vector<int> lob_column_indices_;

  bool closed_;
  int64_t written_rows_;

  // Transaction for manifest management
  std::unique_ptr<api::transaction::Transaction> transaction_;

  // TEXT column writers, keyed by column index
  std::unordered_map<int, std::unique_ptr<lob_column::LobColumnWriter>> lob_writers_;

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

  if (config.lob_base_path.empty() && !config.lob_columns.empty()) {
    return arrow::Status::Invalid("lob_base_path is required when lob_columns is not empty");
  }

  if (config.segment_path.empty()) {
    return arrow::Status::Invalid("segment_path is empty");
  }

  // validate required properties for ColumnGroupPolicy
  if (config.properties.find(PROPERTY_WRITER_POLICY) == config.properties.end()) {
    return arrow::Status::Invalid("properties must contain " + std::string(PROPERTY_WRITER_POLICY));
  }

  // identify TEXT columns and build storage schema
  std::vector<std::shared_ptr<arrow::Field>> storage_fields;
  std::vector<int> lob_column_indices;

  for (int i = 0; i < schema->num_fields(); i++) {
    auto field = schema->field(i);

    if (config.lob_columns.count(GetFieldId(field)) > 0) {
      // TEXT column: convert utf8 to binary for storing LOBReference
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
