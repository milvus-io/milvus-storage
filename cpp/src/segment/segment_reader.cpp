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

#include "milvus-storage/segment/segment_reader.h"

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <unordered_set>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/text_column/text_column_reader.h"
#include "milvus-storage/transaction/transaction.h"

namespace milvus_storage::segment {

// helper function to get field ID from arrow field metadata
static int64_t GetFieldId(const std::shared_ptr<arrow::Field>& field) {
  auto metadata = field->metadata();
  if (!metadata || !metadata->Contains(ARROW_FIELD_ID_KEY)) {
    return -1;
  }
  auto field_id_str = metadata->Get(ARROW_FIELD_ID_KEY).ValueOrDie();
  return std::stoll(field_id_str);
}

// helper function to check if a field is a TEXT column
static bool IsTextField(const std::shared_ptr<arrow::Field>& field,
                        const std::map<int64_t, text_column::TextColumnConfig>& text_columns) {
  auto field_id = GetFieldId(field);
  return text_columns.count(field_id) > 0;
}

// implementation of SegmentReader
class SegmentReaderImpl : public SegmentReader {
  public:
  SegmentReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::shared_ptr<arrow::Schema> original_schema,
                    std::shared_ptr<arrow::Schema> extracted_schema,
                    std::shared_ptr<arrow::Schema> storage_schema,
                    std::vector<std::string> extracted_columns,
                    const SegmentReaderConfig& config,
                    std::vector<int> text_column_indices,
                    int64_t version)
      : fs_(std::move(fs)),
        original_schema_(std::move(original_schema)),
        extracted_schema_(std::move(extracted_schema)),
        storage_schema_(std::move(storage_schema)),
        extracted_columns_(std::move(extracted_columns)),
        config_(config),
        text_column_indices_(std::move(text_column_indices)),
        version_(version),
        closed_(false),
        total_rows_(0) {}

  ~SegmentReaderImpl() override {
    if (!closed_) {
      (void)Close();
    }
  }

  arrow::Status Init(const std::shared_ptr<api::ColumnGroups>& column_groups) {
    column_groups_ = column_groups;

    // create api::Reader with needed columns (storage schema column names)
    std::vector<std::string> storage_column_names;
    for (int i = 0; i < storage_schema_->num_fields(); i++) {
      storage_column_names.push_back(storage_schema_->field(i)->name());
    }

    auto needed_columns = std::make_shared<std::vector<std::string>>(storage_column_names);
    reader_ = api::Reader::create(column_groups_, storage_schema_, needed_columns, config_.properties);

    // create TextColumnReaders for each TEXT column in extracted columns
    for (int extracted_idx : text_column_indices_) {
      auto field = extracted_schema_->field(extracted_idx);
      auto field_id = GetFieldId(field);
      if (field_id < 0) {
        return arrow::Status::Invalid("TEXT column must have a valid field_id in metadata");
      }

      auto it = config_.text_columns.find(field_id);
      if (it == config_.text_columns.end()) {
        return arrow::Status::Invalid("TEXT column config not found for field_id: " + std::to_string(field_id));
      }

      ARROW_ASSIGN_OR_RAISE(auto text_reader, text_column::CreateTextColumnReader(fs_, it->second));
      text_readers_[extracted_idx] = std::move(text_reader);
    }

    // get record batch reader for sequential access
    ARROW_ASSIGN_OR_RAISE(batch_reader_, reader_->get_record_batch_reader());

    return arrow::Status::OK();
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    // read from underlying reader
    std::shared_ptr<arrow::RecordBatch> storage_batch;
    ARROW_RETURN_NOT_OK(batch_reader_->ReadNext(&storage_batch));

    if (!storage_batch) {
      *batch = nullptr;
      return arrow::Status::OK();
    }

    total_rows_ += storage_batch->num_rows();

    // resolve TEXT columns
    ARROW_ASSIGN_OR_RAISE(*batch, ResolveTextColumns(storage_batch));

    return arrow::Status::OK();
  }

  arrow::Result<std::shared_ptr<arrow::Table>> Take(const std::vector<int64_t>& row_indices,
                                                    size_t parallelism) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    if (row_indices.empty()) {
      // return empty table with correct schema
      return arrow::Table::MakeEmpty(extracted_schema_);
    }

    // use api::Reader's take method
    ARROW_ASSIGN_OR_RAISE(auto storage_table, reader_->take(row_indices, parallelism));

    // convert table to batches, resolve TEXT, then convert back
    arrow::TableBatchReader table_reader(*storage_table);
    std::vector<std::shared_ptr<arrow::RecordBatch>> resolved_batches;

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(table_reader.ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_ASSIGN_OR_RAISE(auto resolved_batch, ResolveTextColumns(batch));
      resolved_batches.push_back(resolved_batch);
    }

    if (resolved_batches.empty()) {
      return arrow::Table::MakeEmpty(extracted_schema_);
    }

    return arrow::Table::FromRecordBatches(extracted_schema_, resolved_batches);
  }

  std::shared_ptr<arrow::Schema> schema() const override { return extracted_schema_; }

  std::shared_ptr<arrow::Schema> GetOriginalSchema() const override { return original_schema_; }

  const std::vector<std::string>& GetExtractedColumns() const override { return extracted_columns_; }

  int64_t GetTotalRows() const override { return total_rows_; }

  arrow::Status Close() override {
    if (closed_) {
      return arrow::Status::OK();
    }

    // close TextColumnReaders
    for (auto& [idx, reader] : text_readers_) {
      ARROW_RETURN_NOT_OK(reader->Close());
    }

    closed_ = true;
    return arrow::Status::OK();
  }

  bool IsClosed() const override { return closed_; }

  int64_t GetVersion() const override { return version_; }

  private:
  // resolve TEXT columns from LOBReferences to actual text
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> ResolveTextColumns(
      const std::shared_ptr<arrow::RecordBatch>& storage_batch) {
    if (text_readers_.empty()) {
      // no TEXT columns, return as-is but with extracted schema
      return arrow::RecordBatch::Make(extracted_schema_, storage_batch->num_rows(), storage_batch->columns());
    }

    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve(storage_batch->num_columns());

    for (int i = 0; i < storage_batch->num_columns(); i++) {
      auto it = text_readers_.find(i);
      if (it != text_readers_.end()) {
        // TEXT column: resolve LOBReference to text
        auto ref_array = std::dynamic_pointer_cast<arrow::BinaryArray>(storage_batch->column(i));
        if (!ref_array) {
          return arrow::Status::Invalid("expected BinaryArray for TEXT column reference");
        }
        ARROW_ASSIGN_OR_RAISE(auto text_array, it->second->ReadArrowArray(ref_array));
        columns.push_back(std::static_pointer_cast<arrow::Array>(text_array));
      } else {
        // regular column: use directly
        columns.push_back(storage_batch->column(i));
      }
    }

    return arrow::RecordBatch::Make(extracted_schema_, storage_batch->num_rows(), columns);
  }

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> original_schema_;
  std::shared_ptr<arrow::Schema> extracted_schema_;
  std::shared_ptr<arrow::Schema> storage_schema_;
  std::vector<std::string> extracted_columns_;
  SegmentReaderConfig config_;
  std::vector<int> text_column_indices_;
  int64_t version_;

  bool closed_;
  int64_t total_rows_;

  std::shared_ptr<api::ColumnGroups> column_groups_;
  std::unique_ptr<api::Reader> reader_;
  std::shared_ptr<arrow::RecordBatchReader> batch_reader_;

  // TEXT column readers, keyed by index in extracted schema
  std::map<int, std::unique_ptr<text_column::TextColumnReader>> text_readers_;
};

// helper function to build schemas and identify TEXT columns for extraction
static arrow::Result<std::tuple<std::shared_ptr<arrow::Schema>,  // extracted_schema (user-facing, utf8 for TEXT)
                                std::shared_ptr<arrow::Schema>,  // storage_schema (binary for TEXT)
                                std::vector<int>>>               // text_column_indices in extracted_schema
BuildSchemasForExtraction(const std::shared_ptr<arrow::Schema>& original_schema,
                          const std::vector<std::string>& columns,
                          const std::map<int64_t, text_column::TextColumnConfig>& text_columns) {
  // build column name set for quick lookup
  std::unordered_set<std::string> column_set(columns.begin(), columns.end());
  bool extract_all = columns.empty();

  std::vector<std::shared_ptr<arrow::Field>> extracted_fields;
  std::vector<std::shared_ptr<arrow::Field>> storage_fields;
  std::vector<int> text_column_indices;

  int extracted_idx = 0;
  for (int i = 0; i < original_schema->num_fields(); i++) {
    auto field = original_schema->field(i);

    // check if this column should be extracted
    if (!extract_all && column_set.find(field->name()) == column_set.end()) {
      continue;
    }

    if (IsTextField(field, text_columns)) {
      // TEXT column: extracted schema has utf8, storage schema has binary
      extracted_fields.push_back(field);  // keep original utf8
      auto storage_field = arrow::field(field->name(), arrow::binary(), field->nullable(), field->metadata()->Copy());
      storage_fields.push_back(storage_field);
      text_column_indices.push_back(extracted_idx);
    } else {
      // regular column: same in both schemas
      extracted_fields.push_back(field);
      storage_fields.push_back(field);
    }
    extracted_idx++;
  }

  if (extracted_fields.empty()) {
    return arrow::Status::Invalid("no columns to extract");
  }

  auto extracted_schema = arrow::schema(extracted_fields);
  auto storage_schema = arrow::schema(storage_fields);

  return std::make_tuple(extracted_schema, storage_schema, text_column_indices);
}

// factory function - create from ColumnGroups
arrow::Result<std::unique_ptr<SegmentReader>> SegmentReader::Create(
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::shared_ptr<api::ColumnGroups>& column_groups,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::string>& columns,
    const SegmentReaderConfig& config) {
  if (!fs) {
    return arrow::Status::Invalid("filesystem is null");
  }

  if (!schema) {
    return arrow::Status::Invalid("schema is null");
  }

  if (!column_groups) {
    return arrow::Status::Invalid("column_groups is null");
  }

  // build extracted columns list
  std::vector<std::string> extracted_columns;
  if (columns.empty()) {
    // extract all columns
    for (int i = 0; i < schema->num_fields(); i++) {
      extracted_columns.push_back(schema->field(i)->name());
    }
  } else {
    extracted_columns = columns;
  }

  // build schemas
  ARROW_ASSIGN_OR_RAISE(auto schema_info, BuildSchemasForExtraction(schema, extracted_columns, config.text_columns));
  auto& [extracted_schema, storage_schema, text_column_indices] = schema_info;

  // create implementation (version = -1 when created directly from ColumnGroups)
  auto reader =
      std::make_unique<SegmentReaderImpl>(std::move(fs), schema, extracted_schema, storage_schema,
                                          std::move(extracted_columns), config, std::move(text_column_indices), -1);

  ARROW_RETURN_NOT_OK(reader->Init(column_groups));

  return reader;
}

// factory function - open from manifest
arrow::Result<std::unique_ptr<SegmentReader>> SegmentReader::Open(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                  const std::string& segment_path,
                                                                  int64_t version,
                                                                  const std::shared_ptr<arrow::Schema>& schema,
                                                                  const std::vector<std::string>& columns,
                                                                  const SegmentReaderConfig& config) {
  if (!fs) {
    return arrow::Status::Invalid("filesystem is null");
  }

  if (!schema) {
    return arrow::Status::Invalid("schema is null");
  }

  if (segment_path.empty()) {
    return arrow::Status::Invalid("segment_path is empty");
  }

  // open transaction to read manifest
  ARROW_ASSIGN_OR_RAISE(auto transaction, api::transaction::Transaction::Open(fs, segment_path, version,
                                                                              api::transaction::FailResolver, 1));

  int64_t read_version = transaction->GetReadVersion();

  // get manifest
  ARROW_ASSIGN_OR_RAISE(auto manifest, transaction->GetManifest());

  // extract column groups from manifest
  auto column_groups = std::make_shared<api::ColumnGroups>(manifest->columnGroups());

  // build extracted columns list
  std::vector<std::string> extracted_columns;
  if (columns.empty()) {
    // extract all columns
    for (int i = 0; i < schema->num_fields(); i++) {
      extracted_columns.push_back(schema->field(i)->name());
    }
  } else {
    extracted_columns = columns;
  }

  // build schemas
  ARROW_ASSIGN_OR_RAISE(auto schema_info, BuildSchemasForExtraction(schema, extracted_columns, config.text_columns));
  auto& [extracted_schema, storage_schema, text_column_indices] = schema_info;

  // create implementation
  auto reader = std::make_unique<SegmentReaderImpl>(std::move(fs), schema, extracted_schema, storage_schema,
                                                    std::move(extracted_columns), config,
                                                    std::move(text_column_indices), read_version);

  ARROW_RETURN_NOT_OK(reader->Init(column_groups));

  return reader;
}

}  // namespace milvus_storage::segment
