// Copyright 2023 Zilliz
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

#include "delete_evaluator.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <arrow/compute/api.h>
#include <arrow/compute/expression.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/bit_util.h>
#include <fmt/format.h>

#include "common/sql_predicate_arrow.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::api {

namespace cp = arrow::compute;

// PRIMARY_KEY delta logs identify columns by position: column 0 = primary key
// (int64 or string), column 1 = delete timestamp (int64). Column order is the
// contract (the V3/packed writer does not name them meaningfully).
static constexpr int kPrimaryKeyDeltaPkColumn = 0;
static constexpr int kPrimaryKeyDeltaTsColumn = 1;
// Predicate delta logs are written by Milvus' packed (V3) writer with empty
// logical column names, so columns are identified by position (0 = predicate
// SQL, 1 = delete timestamp) and validated by their mutually-exclusive types.
static constexpr int kPredicateDeltaPredicateColumn = 0;
static constexpr int kPredicateDeltaDeleteTimestampColumn = 1;

// Transparent hash so string primary-key lookups can key on a std::string_view
// (arrow StringArray::GetView) without materializing a std::string per data row.
struct TransparentStringHash {
  using is_transparent = void;
  size_t operator()(std::string_view value) const noexcept { return std::hash<std::string_view>{}(value); }
};

static arrow::Result<std::shared_ptr<arrow::Buffer>> MakeAllTrueBitmap(int64_t length) {
  if (length < 0) {
    return arrow::Status::Invalid("Mask length must be >= 0");
  }
  ARROW_ASSIGN_OR_RAISE(auto values, arrow::AllocateBitmap(length));
  if (length > 0) {
    std::memset(values->mutable_data(), 0xFF, arrow::bit_util::BytesForBits(length));
  }
  return values;
}

static arrow::Result<std::shared_ptr<arrow::BooleanArray>> MakeAllTrueBooleanArray(int64_t length) {
  ARROW_ASSIGN_OR_RAISE(auto values, MakeAllTrueBitmap(length));
  return std::make_shared<arrow::BooleanArray>(length, std::move(values), nullptr, 0);
}

// Marks row `index` deleted in the keep bitmap and decrements `alive_count`, but
// only when the row is still alive. The guard keeps `alive_count` exact when the
// same row is deleted by more than one path (PK + predicate, or multiple
// predicate expressions), which enables an O(1) "all rows deleted" early-out.
static void MarkRowDeleted(uint8_t* mask, int64_t index, int64_t& alive_count) {
  if (arrow::bit_util::GetBit(mask, index)) {
    arrow::bit_util::ClearBit(mask, index);
    --alive_count;
  }
}

static arrow::Result<int> FindFieldIndexByFieldId(const std::shared_ptr<arrow::Schema>& schema, int64_t field_id) {
  if (!schema) {
    return arrow::Status::Invalid("Schema is required to resolve field id ", field_id);
  }
  for (int i = 0; i < schema->num_fields(); ++i) {
    if (milvus_storage::GetFieldId(schema->field(i)) == field_id) {
      return i;
    }
  }
  return -1;
}

class OwningRecordBatchReader final : public arrow::RecordBatchReader {
  public:
  OwningRecordBatchReader(std::shared_ptr<FormatReader> owner, std::shared_ptr<arrow::RecordBatchReader> reader)
      : owner_(std::move(owner)), reader_(std::move(reader)) {}

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override { return reader_->ReadNext(out); }

  [[nodiscard]] std::shared_ptr<arrow::Schema> schema() const override { return reader_->schema(); }

  private:
  std::shared_ptr<FormatReader> owner_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
};

static arrow::Result<std::shared_ptr<FormatReader>> OpenDeltaLogReader(
    const DeltaLog& delta_log,
    const Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ColumnGroupFile file{
      .path = delta_log.path,
      .start_index = 0,
      .end_index = delta_log.num_entries,
      .properties = {},
  };
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, delta_log.path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(delta_log.path));
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(uri.key));
  if (file_info.type() != arrow::fs::FileType::File) {
    return arrow::Status::IOError("Delta log is not a regular file: ", delta_log.path);
  }
  file.Set(kPropertyFileSize, file_info.size());
  const std::vector<std::string> needed_columns;
  return FormatReader::create(nullptr, LOON_FORMAT_PARQUET, file, properties, needed_columns, key_retriever);
}

static arrow::Result<uint64_t> ValidateDeleteTimestamp(int64_t raw_delete_ts, const std::string& path) {
  if (raw_delete_ts < 0) {
    return arrow::Status::Invalid("Delete timestamp must be non-negative: ", path);
  }
  return static_cast<uint64_t>(raw_delete_ts);
}

static arrow::Result<std::string> ResolveTimestampFieldName(const std::shared_ptr<arrow::Schema>& schema,
                                                            int64_t timestamp_field_id) {
  ARROW_ASSIGN_OR_RAISE(auto ts_index, FindFieldIndexByFieldId(schema, timestamp_field_id));
  if (ts_index < 0) {
    return arrow::Status::Invalid("Row timestamp field id ", timestamp_field_id, " not found in schema");
  }
  if (schema->field(ts_index)->type()->id() != arrow::Type::INT64) {
    return arrow::Status::Invalid("Row timestamp field id ", timestamp_field_id, " must be int64");
  }
  return schema->field(ts_index)->name();
}

static arrow::Result<cp::Expression> BuildPredicateDeleteExpression(const std::string& predicate_sql,
                                                                    uint64_t delete_timestamp,
                                                                    const std::shared_ptr<arrow::Schema>& schema,
                                                                    const std::string& timestamp_field_name) {
  ARROW_ASSIGN_OR_RAISE(auto predicate_expr, ParseSqlPredicateToArrowExpression(predicate_sql, schema));
  auto timestamp_expr =
      cp::less_equal(cp::field_ref(timestamp_field_name), cp::literal(static_cast<int64_t>(delete_timestamp)));
  return cp::and_(std::move(predicate_expr), std::move(timestamp_expr));
}

// No reusable storage-side primary-key delete mask helper exists yet. This
// internal evaluator keeps alive stream delete handling limited to manifest-
// derived delete information and storage-owned SQL predicate delete expressions.
class DeleteEvaluator {
  public:
  static arrow::Result<std::shared_ptr<DeleteEvaluator>> Create(
      std::shared_ptr<Manifest> manifest,
      std::shared_ptr<arrow::Schema> schema,
      Properties properties,
      MaskedReadOptions options,
      std::function<std::string(const std::string&)> key_retriever) {
    if (!manifest) {
      return arrow::Status::Invalid("DeleteEvaluator requires manifest");
    }
    auto evaluator = std::shared_ptr<DeleteEvaluator>(new DeleteEvaluator(
        std::move(manifest), std::move(schema), std::move(properties), options, std::move(key_retriever)));
    ARROW_RETURN_NOT_OK(evaluator->LoadDeltaLogs());
    ARROW_RETURN_NOT_OK(evaluator->BuildPredicateExpressions());
    ARROW_RETURN_NOT_OK(evaluator->ResolveNeededColumns());
    return evaluator;
  }

  [[nodiscard]] bool empty() const {
    return int64_pk_delete_ts_.empty() && string_pk_delete_ts_.empty() && predicate_delete_expressions_.empty();
  }

  [[nodiscard]] const std::vector<std::string>& NeededColumns() const { return needed_columns_; }

  arrow::Result<std::shared_ptr<arrow::BooleanArray>> EvaluateKeepMask(
      const std::shared_ptr<arrow::RecordBatch>& batch) const {
    if (!batch) {
      return arrow::Status::Invalid("Cannot evaluate delete mask for null batch");
    }
    if (empty()) {
      return MakeAllTrueBooleanArray(batch->num_rows());
    }

    ARROW_ASSIGN_OR_RAISE(auto mask, MakeAllTrueBitmap(batch->num_rows()));
    // Track the number of still-alive rows so we can stop clearing once every
    // row is deleted (an O(1) check instead of rescanning the bitmap).
    int64_t alive_count = batch->num_rows();
    ARROW_ASSIGN_OR_RAISE(auto timestamp_field_id, RowTimestampFieldId());
    ARROW_ASSIGN_OR_RAISE(auto ts_index, FindRequiredFieldIndex(batch->schema(), timestamp_field_id, "row timestamp"));
    auto ts_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(ts_index));
    if (!ts_array) {
      return arrow::Status::Invalid("Row timestamp column must be int64");
    }

    if (HasPrimaryKeyDeletes()) {
      ARROW_ASSIGN_OR_RAISE(auto pk_index, FindRequiredFieldIndex(batch->schema(), *pk_field_id_, "primary key"));
      switch (batch->column(pk_index)->type_id()) {
        case arrow::Type::INT64:
          ARROW_RETURN_NOT_OK(EvaluateInt64PrimaryKey(batch, pk_index, *ts_array, mask->mutable_data(), alive_count));
          break;
        case arrow::Type::STRING:
          ARROW_RETURN_NOT_OK(EvaluateStringPrimaryKey(batch, pk_index, *ts_array, mask->mutable_data(), alive_count));
          break;
        default:
          return arrow::Status::Invalid(
              fmt::format("Unsupported primary-key column type: {}", batch->column(pk_index)->type()->ToString()));
      }
    }
    // Skip predicate evaluation entirely if primary-key deletes already removed
    // every row.
    if (alive_count > 0 && HasPredicateDeletes()) {
      ARROW_RETURN_NOT_OK(EvaluatePredicateDeletes(batch, mask->mutable_data(), alive_count));
    }

    return std::make_shared<arrow::BooleanArray>(batch->num_rows(), std::move(mask), nullptr, 0);
  }

  private:
  DeleteEvaluator(std::shared_ptr<Manifest> manifest,
                  std::shared_ptr<arrow::Schema> schema,
                  Properties properties,
                  MaskedReadOptions options,
                  std::function<std::string(const std::string&)> key_retriever)
      : manifest_(std::move(manifest)),
        schema_(std::move(schema)),
        properties_(std::move(properties)),
        options_(options),
        key_retriever_(std::move(key_retriever)) {}

  [[nodiscard]] bool HasPrimaryKeyDeletes() const {
    return !int64_pk_delete_ts_.empty() || !string_pk_delete_ts_.empty();
  }

  [[nodiscard]] bool HasPredicateDeletes() const { return !predicate_delete_expressions_.empty(); }

  arrow::Status LoadDeltaLogs() {
    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.num_entries < 0) {
        return arrow::Status::Invalid("Delta log num_entries must be >= 0: ", delta_log.path);
      }
      if (delta_log.num_entries == 0) {
        continue;
      }

      switch (delta_log.type) {
        case DeltaLogType::PRIMARY_KEY:
        case DeltaLogType::PREDICATE:
          break;
        case DeltaLogType::POSITIONAL:
          return arrow::Status::NotImplemented("Unsupported delta log: Positional type for masked reader: ",
                                               delta_log.path);
        default:
          return arrow::Status::Invalid(
              fmt::format("Unknown delta log type {}: {}", static_cast<int>(delta_log.type), delta_log.path));
      }

      ARROW_ASSIGN_OR_RAISE(auto format_reader, OpenDeltaLogReader(delta_log, properties_, key_retriever_));
      ARROW_ASSIGN_OR_RAISE(auto batch_reader, format_reader->read_with_range(0, delta_log.num_entries));
      auto reader = std::make_shared<OwningRecordBatchReader>(std::move(format_reader), std::move(batch_reader));
      switch (delta_log.type) {
        case DeltaLogType::PRIMARY_KEY:
          ARROW_RETURN_NOT_OK(EnsurePrimaryKeyDeleteSupport());
          ARROW_RETURN_NOT_OK(LoadPrimaryKeyDeltaLog(delta_log, std::move(reader)));
          break;
        case DeltaLogType::PREDICATE:
          ARROW_RETURN_NOT_OK(EnsurePredicateDeleteSupport());
          ARROW_RETURN_NOT_OK(LoadPredicateDeltaLog(delta_log, std::move(reader)));
          break;
        case DeltaLogType::POSITIONAL:
        default:
          break;
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status EnsurePrimaryKeyDeleteSupport() {
    if (pk_field_id_.has_value()) {
      return arrow::Status::OK();
    }
    if (!schema_) {
      return arrow::Status::Invalid("PRIMARY_KEY delta logs require an explicit schema with PARQUET:field_id metadata");
    }
    if (!options_.pk_field_id.has_value()) {
      return arrow::Status::Invalid(
          "PRIMARY_KEY delta logs require MaskedReadOptions.pk_field_id: milvus-storage has no primary-key concept, so "
          "the caller must supply the primary-key field id.");
    }
    pk_field_id_ = *options_.pk_field_id;

    ARROW_ASSIGN_OR_RAISE(auto schema_pk_index, FindFieldIndexByFieldId(schema_, *pk_field_id_));
    if (schema_pk_index < 0) {
      return arrow::Status::Invalid(fmt::format("Primary-key field id {} not found in schema", *pk_field_id_));
    }
    ARROW_RETURN_NOT_OK(EnsureRowTimestampField());
    return arrow::Status::OK();
  }

  arrow::Status EnsurePredicateDeleteSupport() {
    if (!schema_) {
      return arrow::Status::Invalid("PREDICATE delta logs require an explicit schema with PARQUET:field_id metadata");
    }
    ARROW_RETURN_NOT_OK(EnsureRowTimestampField());
    return arrow::Status::OK();
  }

  // milvus-storage has no inherent row-timestamp field, so the caller must declare
  // which schema field id carries the per-row timestamp (row_ts <= delete_ts).
  arrow::Result<int64_t> RowTimestampFieldId() const {
    if (!options_.row_timestamp_field_id.has_value()) {
      return arrow::Status::Invalid(
          "Delete handling requires MaskedReadOptions.row_timestamp_field_id: milvus-storage has no inherent "
          "row-timestamp field; the caller must declare it.");
    }
    return *options_.row_timestamp_field_id;
  }

  arrow::Status EnsureRowTimestampField() const {
    ARROW_ASSIGN_OR_RAISE(auto timestamp_field_id, RowTimestampFieldId());
    ARROW_ASSIGN_OR_RAISE(auto timestamp_field_name, ResolveTimestampFieldName(schema_, timestamp_field_id));
    (void)timestamp_field_name;
    return arrow::Status::OK();
  }

  arrow::Status LoadPrimaryKeyDeltaLog(const DeltaLog& delta_log,
                                       const std::shared_ptr<arrow::RecordBatchReader>& reader) {
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      if (batch->num_columns() < 2) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log must contain pk and ts columns: ", delta_log.path);
      }
      // Identify columns purely by position (0 = pk, 1 = delete timestamp), like
      // predicate delta logs; column order is the contract. The type check on the
      // delete-timestamp column catches an obvious misorder for string primary
      // keys (int64 primary keys are indistinguishable from ts by type).
      auto pk_array = batch->column(kPrimaryKeyDeltaPkColumn);
      auto ts_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(kPrimaryKeyDeltaTsColumn));
      if (!ts_array) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log delete timestamp column (1) must be int64: ",
                                      delta_log.path);
      }
      if (pk_array->type_id() == arrow::Type::INT64) {
        auto typed_pk = std::static_pointer_cast<arrow::Int64Array>(pk_array);
        ARROW_RETURN_NOT_OK(LoadInt64PrimaryKeyDeletes(*typed_pk, *ts_array, delta_log.path));
      } else if (pk_array->type_id() == arrow::Type::STRING) {
        auto typed_pk = std::static_pointer_cast<arrow::StringArray>(pk_array);
        ARROW_RETURN_NOT_OK(LoadStringPrimaryKeyDeletes(*typed_pk, *ts_array, delta_log.path));
      } else {
        return arrow::Status::Invalid(fmt::format("Unsupported PRIMARY_KEY delta log pk type '{}' in {}",
                                                  pk_array->type()->ToString(), delta_log.path));
      }
    }
    return reader->Close();
  }

  arrow::Status ValidatePredicateDeltaBatchSchema(const arrow::RecordBatch& batch, const std::string& path) const {
    // V3/packed delta logs carry empty logical column names and identify columns
    // by position: column 0 = predicate SQL (string), column 1 = delete
    // timestamp (int64). The two columns have mutually-exclusive types, so a
    // misordered or malformed file is rejected by the type checks below.
    if (batch.num_columns() < 2) {
      return arrow::Status::Invalid("PREDICATE delta log must contain predicate and delete_timestamp columns: ", path);
    }
    const auto& predicate_field = batch.schema()->field(kPredicateDeltaPredicateColumn);
    if (predicate_field->type()->id() != arrow::Type::STRING) {
      return arrow::Status::Invalid("PREDICATE delta log predicate column (0) must be string: ", path);
    }
    if (predicate_field->nullable()) {
      return arrow::Status::Invalid("PREDICATE delta log predicate column (0) must be non-nullable: ", path);
    }
    const auto& delete_ts_field = batch.schema()->field(kPredicateDeltaDeleteTimestampColumn);
    if (delete_ts_field->type()->id() != arrow::Type::INT64) {
      return arrow::Status::Invalid("PREDICATE delta log delete_timestamp column (1) must be int64: ", path);
    }
    if (delete_ts_field->nullable()) {
      return arrow::Status::Invalid("PREDICATE delta log delete_timestamp column (1) must be non-nullable: ", path);
    }
    return arrow::Status::OK();
  }

  arrow::Result<std::string> GetPredicateSqlValue(const arrow::Array& predicate_sql_array,
                                                  int64_t index,
                                                  const std::string& path) const {
    if (predicate_sql_array.IsNull(index)) {
      return arrow::Status::Invalid("PREDICATE delta log predicate must not contain nulls: ", path);
    }
    if (predicate_sql_array.type_id() != arrow::Type::STRING) {
      return arrow::Status::Invalid("PREDICATE delta log predicate column must be string: ", path);
    }
    return static_cast<const arrow::StringArray&>(predicate_sql_array).GetString(index);
  }

  arrow::Status LoadPredicateDeltaLog(const DeltaLog& delta_log,
                                      const std::shared_ptr<arrow::RecordBatchReader>& reader) {
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(ValidatePredicateDeltaBatchSchema(*batch, delta_log.path));
      auto predicate_sql_array = batch->column(kPredicateDeltaPredicateColumn);
      auto delete_ts_array =
          std::static_pointer_cast<arrow::Int64Array>(batch->column(kPredicateDeltaDeleteTimestampColumn));
      ARROW_RETURN_NOT_OK(AccumulatePredicateDeletes(*delete_ts_array, *predicate_sql_array, delta_log.path));
    }
    return reader->Close();
  }

  // Accumulates predicate delete rows, deduplicating by predicate SQL and keeping
  // the maximum delete timestamp per predicate: a row matching the predicate is
  // deleted iff row_ts <= any of its delete timestamps, i.e. row_ts <= max. This
  // collapses N delete events sharing a predicate into a single expression.
  arrow::Status AccumulatePredicateDeletes(const arrow::Int64Array& delete_ts_array,
                                           const arrow::Array& predicate_sql_array,
                                           const std::string& path) {
    if (delete_ts_array.length() != predicate_sql_array.length()) {
      return arrow::Status::Invalid("PREDICATE delta log column length mismatch: ", path);
    }
    for (int64_t i = 0; i < delete_ts_array.length(); ++i) {
      if (delete_ts_array.IsNull(i)) {
        return arrow::Status::Invalid("PREDICATE delta log delete_timestamp must not contain nulls: ", path);
      }
      ARROW_ASSIGN_OR_RAISE(auto delete_ts, ValidateDeleteTimestamp(delete_ts_array.Value(i), path));
      if (options_.visible_until_ts.has_value() && delete_ts > *options_.visible_until_ts) {
        continue;
      }
      ARROW_ASSIGN_OR_RAISE(auto predicate_sql, GetPredicateSqlValue(predicate_sql_array, i, path));
      auto& max_delete_ts = predicate_max_ts_[predicate_sql];
      max_delete_ts = std::max(max_delete_ts, delete_ts);
    }
    return arrow::Status::OK();
  }

  // Compiles one Arrow expression per unique predicate from the deduplicated map.
  // Parse/bind failures are surfaced here and abort evaluator creation: a
  // predicate that fails to compile is a correctness error, never skipped.
  arrow::Status BuildPredicateExpressions() {
    if (predicate_max_ts_.empty()) {
      return arrow::Status::OK();
    }
    ARROW_ASSIGN_OR_RAISE(auto timestamp_field_id, RowTimestampFieldId());
    ARROW_ASSIGN_OR_RAISE(auto timestamp_field_name, ResolveTimestampFieldName(schema_, timestamp_field_id));
    predicate_delete_expressions_.reserve(predicate_max_ts_.size());
    for (const auto& [predicate_sql, max_delete_ts] : predicate_max_ts_) {
      ARROW_ASSIGN_OR_RAISE(auto delete_expr, BuildPredicateDeleteExpression(predicate_sql, max_delete_ts, schema_,
                                                                             timestamp_field_name));
      ARROW_ASSIGN_OR_RAISE(auto bound_expr, delete_expr.Bind(*schema_));
      (void)bound_expr;
      predicate_delete_expressions_.push_back(std::move(delete_expr));
    }
    return arrow::Status::OK();
  }

  arrow::Status LoadInt64PrimaryKeyDeletes(const arrow::Int64Array& pk_array,
                                           const arrow::Int64Array& ts_array,
                                           const std::string& path) {
    if (pk_array.length() != ts_array.length()) {
      return arrow::Status::Invalid("Delta pk and ts column length mismatch: ", path);
    }
    for (int64_t i = 0; i < pk_array.length(); ++i) {
      if (pk_array.IsNull(i) || ts_array.IsNull(i)) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log pk/ts must not contain nulls: ", path);
      }
      ARROW_ASSIGN_OR_RAISE(auto delete_ts, ValidateDeleteTimestamp(ts_array.Value(i), path));
      if (options_.visible_until_ts.has_value() && delete_ts > *options_.visible_until_ts) {
        continue;
      }
      auto& old_ts = int64_pk_delete_ts_[pk_array.Value(i)];
      old_ts = std::max(old_ts, delete_ts);
    }
    return arrow::Status::OK();
  }

  arrow::Status LoadStringPrimaryKeyDeletes(const arrow::StringArray& pk_array,
                                            const arrow::Int64Array& ts_array,
                                            const std::string& path) {
    if (pk_array.length() != ts_array.length()) {
      return arrow::Status::Invalid("Delta pk and ts column length mismatch: ", path);
    }
    for (int64_t i = 0; i < pk_array.length(); ++i) {
      if (pk_array.IsNull(i) || ts_array.IsNull(i)) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log pk/ts must not contain nulls: ", path);
      }
      ARROW_ASSIGN_OR_RAISE(auto delete_ts, ValidateDeleteTimestamp(ts_array.Value(i), path));
      if (options_.visible_until_ts.has_value() && delete_ts > *options_.visible_until_ts) {
        continue;
      }
      auto& old_ts = string_pk_delete_ts_[pk_array.GetString(i)];
      old_ts = std::max(old_ts, delete_ts);
    }
    return arrow::Status::OK();
  }

  arrow::Status ResolveNeededColumns() {
    needed_columns_.clear();
    std::unordered_set<std::string> seen;
    auto add_column_name = [&](const std::string& name, const std::string& role) -> arrow::Status {
      if (schema_ && schema_->GetFieldIndex(name) < 0) {
        return arrow::Status::Invalid(fmt::format("{} column '{}' not found in schema", role, name));
      }
      if (seen.insert(name).second) {
        needed_columns_.push_back(name);
      }
      return arrow::Status::OK();
    };
    auto add_field_id = [&](int64_t field_id, const std::string& role) -> arrow::Status {
      ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(schema_, field_id));
      if (index < 0) {
        return arrow::Status::Invalid(fmt::format("{} field id {} not found in schema", role, field_id));
      }
      return add_column_name(schema_->field(index)->name(), role);
    };

    if (HasPrimaryKeyDeletes()) {
      if (!pk_field_id_.has_value()) {
        return arrow::Status::Invalid("PRIMARY_KEY delta logs require primary-key field id");
      }
      ARROW_RETURN_NOT_OK(add_field_id(*pk_field_id_, "Primary-key"));
    }
    if (HasPrimaryKeyDeletes() || HasPredicateDeletes()) {
      ARROW_ASSIGN_OR_RAISE(auto timestamp_field_id, RowTimestampFieldId());
      ARROW_RETURN_NOT_OK(add_field_id(timestamp_field_id, "Row timestamp"));
    }
    for (const auto& expression : predicate_delete_expressions_) {
      for (const auto& field_ref : cp::FieldsInExpression(expression)) {
        const auto* name = field_ref.name();
        if (name == nullptr) {
          return arrow::Status::NotImplemented(
              "Only by-name field references are supported in predicate delete expressions: ", field_ref.ToString());
        }
        ARROW_RETURN_NOT_OK(add_column_name(*name, "Predicate delete"));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Result<int> FindRequiredFieldIndex(const std::shared_ptr<arrow::Schema>& schema,
                                            int64_t field_id,
                                            const std::string& role) const {
    ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(schema, field_id));
    if (index < 0) {
      return arrow::Status::Invalid(
          fmt::format("{} field id {} must be included in masked reader output schema", role, field_id));
    }
    return index;
  }

  arrow::Status EvaluateInt64PrimaryKey(const std::shared_ptr<arrow::RecordBatch>& batch,
                                        int pk_index,
                                        const arrow::Int64Array& ts_array,
                                        uint8_t* mask,
                                        int64_t& alive_count) const {
    auto pk_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(pk_index));
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (pk_array->IsNull(i) || ts_array.IsNull(i)) {
        continue;
      }
      auto it = int64_pk_delete_ts_.find(pk_array->Value(i));
      if (it != int64_pk_delete_ts_.end() && static_cast<uint64_t>(ts_array.Value(i)) <= it->second) {
        MarkRowDeleted(mask, i, alive_count);
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status EvaluateStringPrimaryKey(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         int pk_index,
                                         const arrow::Int64Array& ts_array,
                                         uint8_t* mask,
                                         int64_t& alive_count) const {
    auto pk_array = std::static_pointer_cast<arrow::StringArray>(batch->column(pk_index));
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (pk_array->IsNull(i) || ts_array.IsNull(i)) {
        continue;
      }
      auto it = string_pk_delete_ts_.find(pk_array->GetView(i));
      if (it != string_pk_delete_ts_.end() && static_cast<uint64_t>(ts_array.Value(i)) <= it->second) {
        MarkRowDeleted(mask, i, alive_count);
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status EvaluatePredicateDeletes(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         uint8_t* mask,
                                         int64_t& alive_count) const {
    ARROW_ASSIGN_OR_RAISE(auto exec_batch, cp::MakeExecBatch(*batch->schema(), arrow::Datum(batch)));
    // All batches from the reader share the same (projected) schema, so bind the
    // predicate expressions once against the first batch and cache them; the
    // per-batch rebind is otherwise pure repeated work. (The load-time bind
    // against the full schema is validation-only and is not reusable here.)
    if (bound_predicate_expressions_.empty()) {
      bound_predicate_expressions_.reserve(predicate_delete_expressions_.size());
      for (const auto& expression : predicate_delete_expressions_) {
        ARROW_ASSIGN_OR_RAISE(auto bound_expr, expression.Bind(*batch->schema()));
        bound_predicate_expressions_.push_back(std::move(bound_expr));
      }
    }
    for (const auto& bound_expr : bound_predicate_expressions_) {
      // Every remaining row is already deleted; the remaining expressions cannot
      // change the mask, so stop before their (expensive) execution.
      if (alive_count == 0) {
        break;
      }
      ARROW_ASSIGN_OR_RAISE(auto result_datum, cp::ExecuteScalarExpression(bound_expr, exec_batch));
      if (!result_datum.is_array()) {
        return arrow::Status::Invalid("Predicate delete expression must produce a boolean array");
      }
      auto result_array = result_datum.make_array();
      if (result_array->type_id() != arrow::Type::BOOL) {
        return arrow::Status::Invalid("Predicate delete expression must produce boolean values, got ",
                                      result_array->type()->ToString());
      }
      if (result_array->length() != batch->num_rows()) {
        return arrow::Status::Invalid("Predicate delete expression result length mismatch");
      }
      const auto& delete_mask = *std::static_pointer_cast<arrow::BooleanArray>(result_array);
      for (int64_t i = 0; i < delete_mask.length(); ++i) {
        if (!delete_mask.IsNull(i) && delete_mask.Value(i)) {
          MarkRowDeleted(mask, i, alive_count);
        }
      }
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<arrow::Schema> schema_;
  Properties properties_;
  MaskedReadOptions options_;
  std::function<std::string(const std::string&)> key_retriever_;
  std::optional<int64_t> pk_field_id_;
  std::unordered_map<int64_t, uint64_t> int64_pk_delete_ts_;
  std::unordered_map<std::string, uint64_t, TransparentStringHash, std::equal_to<>> string_pk_delete_ts_;
  // predicate SQL -> max delete timestamp, deduplicated at load time; compiled
  // into predicate_delete_expressions_ (one per unique predicate) after loading.
  std::unordered_map<std::string, uint64_t> predicate_max_ts_;
  std::vector<cp::Expression> predicate_delete_expressions_;
  // Predicate expressions bound to the (stable) evaluation batch schema, cached
  // on the first batch to avoid re-binding on every batch.
  mutable std::vector<cp::Expression> bound_predicate_expressions_;
  std::vector<std::string> needed_columns_;
};

arrow::Result<std::shared_ptr<DeleteEvaluator>> CreateDeleteEvaluator(
    std::shared_ptr<Manifest> manifest,
    std::shared_ptr<arrow::Schema> schema,
    Properties properties,
    MaskedReadOptions options,
    std::function<std::string(const std::string&)> key_retriever) {
  return DeleteEvaluator::Create(std::move(manifest), std::move(schema), std::move(properties), options,
                                 std::move(key_retriever));
}

const std::vector<std::string>& DeleteEvaluatorNeededColumns(const std::shared_ptr<DeleteEvaluator>& evaluator) {
  return evaluator->NeededColumns();
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> EvaluateDeleteKeepMask(
    const std::shared_ptr<DeleteEvaluator>& evaluator, const std::shared_ptr<arrow::RecordBatch>& batch) {
  return evaluator->EvaluateKeepMask(batch);
}

}  // namespace milvus_storage::api
