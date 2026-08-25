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

#pragma once
#include <charconv>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <arrow/table.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>
#include <folly/futures/Future.h>
#include <parquet/arrow/reader.h>
#include <parquet/properties.h>
#include <arrow/filesystem/filesystem.h>

#include "milvus-storage/common/constants.h"

namespace milvus_storage {

namespace detail {

// Converts an Arrow status into a ready error future for any
// folly::SemiFuture<arrow::Result<T>> return type.
class FollyArrowErrorFuture {
  public:
  // Store the setup failure until the enclosing function's Result<T> is inferred.
  explicit FollyArrowErrorFuture(arrow::Status status) : status_(std::move(status)) {}

  // Materialize a ready error future without requiring callers to spell T.
  template <typename T>
  operator folly::SemiFuture<arrow::Result<T>>() && {
    return folly::makeSemiFuture(arrow::Result<T>(std::move(status_)));
  }

  private:
  arrow::Status status_;
};

}  // namespace detail

// Validation-safe field-id parsing. Authoritative PARQUET:field_id metadata is
// parsed strictly and returned as a status instead of throwing. When metadata
// is absent, keep the legacy field-name fallback unchanged for compatibility.
inline arrow::Result<int64_t> TryGetFieldId(const std::shared_ptr<arrow::Field>& field) {
  if (!field) {
    return arrow::Status::Invalid("Arrow field must not be null");
  }

  auto metadata = field->metadata();
  if (metadata && metadata->Contains(ARROW_FIELD_ID_KEY)) {
    ARROW_ASSIGN_OR_RAISE(auto value, metadata->Get(ARROW_FIELD_ID_KEY));
    int64_t field_id = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, field_id);
    if (ec != std::errc{} || ptr != end || field_id < 0) {
      return arrow::Status::Invalid("Invalid ", ARROW_FIELD_ID_KEY, " value '", value, "' for field '", field->name(),
                                    "': expected a non-negative int64");
    }
    return field_id;
  }

  // A missing field-id is a valid schema shape for non-packed callers. Keep
  // the historical sentinel, but do not accept numeric prefixes: only a
  // complete, non-negative decimal name is an id.
  const auto& name = field->name();
  int64_t field_id = 0;
  const auto* begin = name.data();
  const auto* end = begin + name.size();
  const auto [ptr, ec] = std::from_chars(begin, end, field_id);
  if (ec != std::errc{} || ptr != end || field_id < 0) {
    return -1;
  }
  return field_id;
}

// Extract field_id from Arrow field.
//
// Keep the original return type and parsing behavior because this helper is in
// an installed public header. New validation-sensitive code should use
// TryGetFieldId instead.
inline int64_t GetFieldId(const std::shared_ptr<arrow::Field>& field) {
  auto metadata = field->metadata();
  if (metadata && metadata->Contains(ARROW_FIELD_ID_KEY)) {
    return std::stoll(metadata->Get(ARROW_FIELD_ID_KEY).ValueOrDie());
  }
  try {
    return std::stoll(field->name());
  } catch (const std::invalid_argument&) {
    return -1;
  } catch (const std::out_of_range&) {
    return -1;
  }
}

inline arrow::Status ValidateFieldIds(const std::shared_ptr<arrow::Schema>& schema) {
  if (!schema) {
    return arrow::Status::Invalid("Arrow schema must not be null");
  }
  for (const auto& field : schema->fields()) {
    auto field_id = TryGetFieldId(field);
    ARROW_RETURN_NOT_OK(field_id.status());
  }
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<::parquet::arrow::FileReader>> MakeArrowFileReader(
    arrow::fs::FileSystem& fs,
    const std::string& file_path,
    const ::parquet::ReaderProperties& read_properties,
    const ::parquet::ArrowReaderProperties& arrow_reader_properties);

size_t GetRecordBatchMemorySize(const std::shared_ptr<arrow::RecordBatch>& record_batch);

size_t GetArrowArrayMemorySize(const std::shared_ptr<arrow::Array>& array);

size_t GetTableMemorySize(const std::shared_ptr<arrow::Table>& table);

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ConvertTableToRecordBatch(const std::shared_ptr<arrow::Table>& table,
                                                                             bool allow_concat = false);
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ConvertTableToRecordBatchs(
    const std::shared_ptr<arrow::Table>& table);

arrow::Result<std::shared_ptr<arrow::Table>> CopySelectedRows(const std::shared_ptr<arrow::Table>& table,
                                                              const std::vector<int64_t>& indices);

arrow::Result<std::string> GetEnvVar(const char* name);

arrow::Result<std::string> GetEnvVar(const std::string& name);

}  // namespace milvus_storage

// Async counterparts of Arrow's early-return helpers. They keep validation and
// setup failures on the same Result-bearing future path as backend failures.
#define FOLLY_ARROW_RETURN_NOT_OK(status_expr)                                                \
  do {                                                                                        \
    auto _folly_arrow_status = (status_expr);                                                 \
    if (!_folly_arrow_status.ok()) {                                                          \
      return ::milvus_storage::detail::FollyArrowErrorFuture(std::move(_folly_arrow_status)); \
    }                                                                                         \
  } while (false)

#define FOLLY_ARROW_ASSIGN_OR_RAISE_NAME_IMPL(x, y) x##y
#define FOLLY_ARROW_ASSIGN_OR_RAISE_NAME(x, y) FOLLY_ARROW_ASSIGN_OR_RAISE_NAME_IMPL(x, y)

#define FOLLY_ARROW_ASSIGN_OR_RAISE_IMPL(result_name, lhs, rexpr)                   \
  auto&& result_name = (rexpr);                                                     \
  if (!(result_name).ok()) {                                                        \
    return ::milvus_storage::detail::FollyArrowErrorFuture((result_name).status()); \
  }                                                                                 \
  lhs = std::move(result_name).ValueUnsafe();

#define FOLLY_ARROW_ASSIGN_OR_RAISE(lhs, rexpr) \
  FOLLY_ARROW_ASSIGN_OR_RAISE_IMPL(FOLLY_ARROW_ASSIGN_OR_RAISE_NAME(_folly_arrow_result, __COUNTER__), lhs, rexpr)
