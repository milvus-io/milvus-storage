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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/bit_util.h>
#include <fmt/format.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader.h"
#include "proto/milvus/plan.pb.h"

namespace milvus_storage {
namespace api {

static constexpr int64_t kMilvusTimestampFieldId = 1;
static constexpr const char* kPrimaryKeyStatsPrefix = "bloom_filter.";
static constexpr const char* kDeltaPkColumnName = "pk";
static constexpr const char* kDeltaTsColumnName = "ts";

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

static arrow::Result<std::optional<int64_t>> ParsePrimaryKeyFieldIdFromStatsKey(const std::string& key) {
  std::string_view view(key);
  std::string_view prefix(kPrimaryKeyStatsPrefix);
  if (view.substr(0, prefix.size()) != prefix || view.size() == prefix.size()) {
    return std::nullopt;
  }
  try {
    return std::stoll(std::string(view.substr(prefix.size())));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Invalid primary-key stats key '{}': {}", key, e.what()));
  }
}

static arrow::Result<std::optional<int64_t>> ResolvePrimaryKeyFieldId(const Manifest& manifest,
                                                                      const std::shared_ptr<arrow::Schema>& schema) {
  std::optional<int64_t> pk_field_id;
  for (const auto& [key, _] : manifest.stats()) {
    ARROW_ASSIGN_OR_RAISE(auto parsed, ParsePrimaryKeyFieldIdFromStatsKey(key));
    if (!parsed.has_value()) {
      continue;
    }
    if (pk_field_id.has_value() && *pk_field_id != *parsed) {
      return arrow::Status::Invalid(
          fmt::format("Multiple primary-key stats keys found: {} and {}", *pk_field_id, *parsed));
    }
    pk_field_id = *parsed;
  }
  if (pk_field_id.has_value()) {
    return pk_field_id;
  }

  return std::nullopt;
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

static arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> OpenDeltaLogReader(
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
  ARROW_ASSIGN_OR_RAISE(auto format_reader, FormatReader::create(nullptr, LOON_FORMAT_PARQUET, file, properties,
                                                                 needed_columns, key_retriever));
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, format_reader->read_with_range(0, delta_log.num_entries));
  return std::make_shared<OwningRecordBatchReader>(std::move(format_reader), std::move(batch_reader));
}


static constexpr const char* kPredicateDeltaFormatVersionKey = "milvus.predicate_delta.format_version";
static constexpr const char* kPredicateDeltaDeleteTsColumnName = "delete_timestamp";
static constexpr const char* kPredicateDeltaSerializedPlanColumnName = "serialized_expr_plan";

enum class PredicateNodeKind {
  kCompare,
  kTerm,
  kAnd,
  kOr,
  kNot,
  kIsNull,
  kIsNotNull,
  kAlwaysTrue,
};

enum class PredicateCompareOp {
  kGreaterThan,
  kGreaterEqual,
  kLessThan,
  kLessEqual,
  kEqual,
  kNotEqual,
};

enum class PredicateValueKind {
  kBool,
  kInt64,
  kDouble,
  kString,
};

struct PredicateValue {
  PredicateValueKind kind;
  bool bool_value = false;
  int64_t int64_value = 0;
  double double_value = 0;
  std::string string_value;
};

struct PredicateNode {
  PredicateNodeKind kind;
  int64_t field_id = -1;
  PredicateCompareOp compare_op = PredicateCompareOp::kEqual;
  PredicateValue value{PredicateValueKind::kBool, true, 0, 0, {}};
  std::vector<PredicateValue> values;
  std::shared_ptr<PredicateNode> left;
  std::shared_ptr<PredicateNode> right;
  std::shared_ptr<PredicateNode> child;
};

enum class PredicateBool {
  kFalse,
  kTrue,
  kNull,
};

struct ParsedPredicatePlan {
  std::shared_ptr<PredicateNode> root;
  std::vector<int64_t> referenced_field_ids;
};

namespace planpb = ::milvus::proto::plan;

static arrow::Result<uint64_t> ValidateDeleteTimestamp(int64_t raw_delete_ts, const std::string& path) {
  if (raw_delete_ts < 0) {
    return arrow::Status::Invalid("Delete timestamp must be non-negative: ", path);
  }
  return static_cast<uint64_t>(raw_delete_ts);
}

static arrow::Result<PredicateValue> ParsePredicateValue(const planpb::GenericValue& value) {
  switch (value.val_case()) {
    case planpb::GenericValue::kBoolVal:
      return PredicateValue{PredicateValueKind::kBool, value.bool_val(), 0, 0, {}};
    case planpb::GenericValue::kInt64Val:
      return PredicateValue{PredicateValueKind::kInt64, false, value.int64_val(), 0, {}};
    case planpb::GenericValue::kFloatVal:
      return PredicateValue{PredicateValueKind::kDouble, false, 0, value.float_val(), {}};
    case planpb::GenericValue::kStringVal:
      return PredicateValue{PredicateValueKind::kString, false, 0, 0, value.string_val()};
    case planpb::GenericValue::kArrayVal:
      return arrow::Status::Invalid("Unsupported predicate GenericValue array_val");
    case planpb::GenericValue::VAL_NOT_SET:
      return arrow::Status::Invalid("Predicate GenericValue missing value");
  }
  return arrow::Status::Invalid("Unknown predicate GenericValue case: ", static_cast<int>(value.val_case()));
}

static arrow::Result<PredicateCompareOp> ParsePredicateCompareOp(planpb::OpType op) {
  switch (op) {
    case planpb::GreaterThan:
      return PredicateCompareOp::kGreaterThan;
    case planpb::GreaterEqual:
      return PredicateCompareOp::kGreaterEqual;
    case planpb::LessThan:
      return PredicateCompareOp::kLessThan;
    case planpb::LessEqual:
      return PredicateCompareOp::kLessEqual;
    case planpb::Equal:
      return PredicateCompareOp::kEqual;
    case planpb::NotEqual:
      return PredicateCompareOp::kNotEqual;
    default:
      return arrow::Status::Invalid("Unsupported predicate UnaryRangeExpr op: ", planpb::OpType_Name(op));
  }
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParsePredicateExpr(const planpb::Expr& expr,
                                                                       std::unordered_set<int64_t>* field_ids);

static arrow::Result<std::shared_ptr<PredicateNode>> ParseUnaryRangeExpr(const planpb::UnaryRangeExpr& expr,
                                                                       std::unordered_set<int64_t>* field_ids) {
  if (!expr.has_column_info()) {
    return arrow::Status::Invalid("UnaryRangeExpr requires column_info");
  }
  if (!expr.has_value()) {
    return arrow::Status::Invalid("UnaryRangeExpr requires value");
  }

  ARROW_ASSIGN_OR_RAISE(auto op, ParsePredicateCompareOp(expr.op()));
  ARROW_ASSIGN_OR_RAISE(auto value, ParsePredicateValue(expr.value()));

  auto node = std::make_shared<PredicateNode>();
  node->kind = PredicateNodeKind::kCompare;
  node->field_id = expr.column_info().field_id();
  field_ids->insert(node->field_id);
  node->compare_op = op;
  node->value = std::move(value);
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParseTermExpr(const planpb::TermExpr& expr,
                                                               std::unordered_set<int64_t>* field_ids) {
  if (!expr.has_column_info()) {
    return arrow::Status::Invalid("TermExpr requires column_info");
  }
  if (expr.values_size() == 0) {
    return arrow::Status::Invalid("TermExpr requires at least one value");
  }
  if (expr.is_in_field()) {
    return arrow::Status::Invalid("TermExpr with is_in_field=true is not supported");
  }

  auto node = std::make_shared<PredicateNode>();
  node->kind = PredicateNodeKind::kTerm;
  node->field_id = expr.column_info().field_id();
  field_ids->insert(node->field_id);
  node->values.reserve(static_cast<size_t>(expr.values_size()));
  for (const auto& raw_value : expr.values()) {
    ARROW_ASSIGN_OR_RAISE(auto value, ParsePredicateValue(raw_value));
    node->values.push_back(std::move(value));
  }
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParseBinaryRangeExpr(const planpb::BinaryRangeExpr& expr,
                                                                      std::unordered_set<int64_t>* field_ids) {
  if (!expr.has_column_info()) {
    return arrow::Status::Invalid("BinaryRangeExpr requires column_info");
  }
  if (!expr.has_lower_value() || !expr.has_upper_value()) {
    return arrow::Status::Invalid("BinaryRangeExpr requires lower_value and upper_value");
  }

  ARROW_ASSIGN_OR_RAISE(auto lower_value, ParsePredicateValue(expr.lower_value()));
  ARROW_ASSIGN_OR_RAISE(auto upper_value, ParsePredicateValue(expr.upper_value()));

  auto lower = std::make_shared<PredicateNode>();
  lower->kind = PredicateNodeKind::kCompare;
  lower->field_id = expr.column_info().field_id();
  field_ids->insert(lower->field_id);
  lower->compare_op = expr.lower_inclusive() ? PredicateCompareOp::kGreaterEqual : PredicateCompareOp::kGreaterThan;
  lower->value = std::move(lower_value);

  auto upper = std::make_shared<PredicateNode>();
  upper->kind = PredicateNodeKind::kCompare;
  upper->field_id = expr.column_info().field_id();
  upper->compare_op = expr.upper_inclusive() ? PredicateCompareOp::kLessEqual : PredicateCompareOp::kLessThan;
  upper->value = std::move(upper_value);

  auto node = std::make_shared<PredicateNode>();
  node->kind = PredicateNodeKind::kAnd;
  node->left = std::move(lower);
  node->right = std::move(upper);
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParseBinaryExpr(const planpb::BinaryExpr& expr,
                                                                 std::unordered_set<int64_t>* field_ids) {
  if (!expr.has_left() || !expr.has_right()) {
    return arrow::Status::Invalid("BinaryExpr requires left and right");
  }

  auto node = std::make_shared<PredicateNode>();
  switch (expr.op()) {
    case planpb::BinaryExpr::LogicalAnd:
      node->kind = PredicateNodeKind::kAnd;
      break;
    case planpb::BinaryExpr::LogicalOr:
      node->kind = PredicateNodeKind::kOr;
      break;
    default:
      return arrow::Status::Invalid("Unsupported BinaryExpr op: ", planpb::BinaryExpr_BinaryOp_Name(expr.op()));
  }

  ARROW_ASSIGN_OR_RAISE(node->left, ParsePredicateExpr(expr.left(), field_ids));
  ARROW_ASSIGN_OR_RAISE(node->right, ParsePredicateExpr(expr.right(), field_ids));
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParseUnaryExpr(const planpb::UnaryExpr& expr,
                                                                std::unordered_set<int64_t>* field_ids) {
  if (expr.op() != planpb::UnaryExpr::Not || !expr.has_child()) {
    return arrow::Status::Invalid("UnaryExpr supports only NOT with a child expression");
  }

  auto node = std::make_shared<PredicateNode>();
  node->kind = PredicateNodeKind::kNot;
  ARROW_ASSIGN_OR_RAISE(node->child, ParsePredicateExpr(expr.child(), field_ids));
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParseNullExpr(const planpb::NullExpr& expr,
                                                               std::unordered_set<int64_t>* field_ids) {
  if (!expr.has_column_info()) {
    return arrow::Status::Invalid("NullExpr requires column_info");
  }

  auto node = std::make_shared<PredicateNode>();
  switch (expr.op()) {
    case planpb::NullExpr::IsNull:
      node->kind = PredicateNodeKind::kIsNull;
      break;
    case planpb::NullExpr::IsNotNull:
      node->kind = PredicateNodeKind::kIsNotNull;
      break;
    default:
      return arrow::Status::Invalid("Unsupported NullExpr op: ", planpb::NullExpr_NullOp_Name(expr.op()));
  }
  node->field_id = expr.column_info().field_id();
  field_ids->insert(node->field_id);
  return node;
}

static arrow::Result<std::shared_ptr<PredicateNode>> ParsePredicateExpr(const planpb::Expr& expr,
                                                                        std::unordered_set<int64_t>* field_ids) {
  switch (expr.expr_case()) {
    case planpb::Expr::kTermExpr:
      return ParseTermExpr(expr.term_expr(), field_ids);
    case planpb::Expr::kUnaryExpr:
      return ParseUnaryExpr(expr.unary_expr(), field_ids);
    case planpb::Expr::kBinaryExpr:
      return ParseBinaryExpr(expr.binary_expr(), field_ids);
    case planpb::Expr::kUnaryRangeExpr:
      return ParseUnaryRangeExpr(expr.unary_range_expr(), field_ids);
    case planpb::Expr::kBinaryRangeExpr:
      return ParseBinaryRangeExpr(expr.binary_range_expr(), field_ids);
    case planpb::Expr::kAlwaysTrueExpr: {
      auto node = std::make_shared<PredicateNode>();
      node->kind = PredicateNodeKind::kAlwaysTrue;
      return node;
    }
    case planpb::Expr::kNullExpr:
      return ParseNullExpr(expr.null_expr(), field_ids);
    case planpb::Expr::EXPR_NOT_SET:
      return arrow::Status::Invalid("Predicate Expr has no expression node");
    default:
      return arrow::Status::Invalid("Unsupported predicate Expr case: ", static_cast<int>(expr.expr_case()));
  }
}

static arrow::Result<ParsedPredicatePlan> ParsePredicatePlan(std::string_view bytes) {
  planpb::PlanNode plan;
  if (!plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
    return arrow::Status::Invalid("Failed to parse serialized predicate PlanNode");
  }

  std::unordered_set<int64_t> field_ids;
  std::shared_ptr<PredicateNode> root;
  if (plan.has_query()) {
    if (!plan.query().has_predicates()) {
      return arrow::Status::Invalid("QueryPlanNode missing predicates");
    }
    ARROW_ASSIGN_OR_RAISE(root, ParsePredicateExpr(plan.query().predicates(), &field_ids));
  } else if (plan.has_predicates()) {
    ARROW_ASSIGN_OR_RAISE(root, ParsePredicateExpr(plan.predicates(), &field_ids));
  } else {
    return arrow::Status::Invalid("PlanNode missing predicate expression");
  }

  std::vector<int64_t> referenced_field_ids(field_ids.begin(), field_ids.end());
  std::sort(referenced_field_ids.begin(), referenced_field_ids.end());
  return ParsedPredicatePlan{std::move(root), std::move(referenced_field_ids)};
}

static PredicateBool PredicateAnd(PredicateBool left, PredicateBool right) {
  if (left == PredicateBool::kFalse || right == PredicateBool::kFalse) {
    return PredicateBool::kFalse;
  }
  if (left == PredicateBool::kTrue && right == PredicateBool::kTrue) {
    return PredicateBool::kTrue;
  }
  return PredicateBool::kNull;
}

static PredicateBool PredicateOr(PredicateBool left, PredicateBool right) {
  if (left == PredicateBool::kTrue || right == PredicateBool::kTrue) {
    return PredicateBool::kTrue;
  }
  if (left == PredicateBool::kFalse && right == PredicateBool::kFalse) {
    return PredicateBool::kFalse;
  }
  return PredicateBool::kNull;
}

static PredicateBool PredicateNot(PredicateBool value) {
  if (value == PredicateBool::kTrue) {
    return PredicateBool::kFalse;
  }
  if (value == PredicateBool::kFalse) {
    return PredicateBool::kTrue;
  }
  return PredicateBool::kNull;
}

template <typename T>
static bool CompareValues(T left, T right, PredicateCompareOp op) {
  switch (op) {
    case PredicateCompareOp::kGreaterThan:
      return left > right;
    case PredicateCompareOp::kGreaterEqual:
      return left >= right;
    case PredicateCompareOp::kLessThan:
      return left < right;
    case PredicateCompareOp::kLessEqual:
      return left <= right;
    case PredicateCompareOp::kEqual:
      return left == right;
    case PredicateCompareOp::kNotEqual:
      return left != right;
  }
  return false;
}

static arrow::Result<PredicateBool> CompareIntegralValue(int64_t left,
                                                                const PredicateValue& right,
                                                                PredicateCompareOp op) {
  if (right.kind == PredicateValueKind::kInt64) {
    return CompareValues(left, right.int64_value, op) ? PredicateBool::kTrue : PredicateBool::kFalse;
  }
  if (right.kind == PredicateValueKind::kDouble) {
    return CompareValues(static_cast<double>(left), right.double_value, op) ? PredicateBool::kTrue : PredicateBool::kFalse;
  }
  return arrow::Status::Invalid("Integer predicate comparison requires numeric literal");
}

static arrow::Result<PredicateBool> CompareFloatingValue(double left,
                                                        const PredicateValue& right,
                                                        PredicateCompareOp op) {
  double rhs = 0;
  if (right.kind == PredicateValueKind::kInt64) {
    rhs = static_cast<double>(right.int64_value);
  } else if (right.kind == PredicateValueKind::kDouble) {
    rhs = right.double_value;
  } else {
    return arrow::Status::Invalid("Floating predicate comparison requires numeric literal");
  }
  return CompareValues(left, rhs, op) ? PredicateBool::kTrue : PredicateBool::kFalse;
}

static arrow::Result<PredicateBool> CompareStringValue(std::string_view left,
                                                       const PredicateValue& right,
                                                       PredicateCompareOp op) {
  if (right.kind != PredicateValueKind::kString) {
    return arrow::Status::Invalid("String predicate comparison requires string literal");
  }
  return CompareValues(left, std::string_view(right.string_value), op) ? PredicateBool::kTrue : PredicateBool::kFalse;
}

static arrow::Result<PredicateBool> CompareBoolValue(bool left, const PredicateValue& right, PredicateCompareOp op) {
  if (right.kind != PredicateValueKind::kBool) {
    return arrow::Status::Invalid("Boolean predicate comparison requires bool literal");
  }
  if (op != PredicateCompareOp::kEqual && op != PredicateCompareOp::kNotEqual) {
    return arrow::Status::Invalid("Boolean predicate comparison supports only equality operators");
  }
  return CompareValues(left, right.bool_value, op) ? PredicateBool::kTrue : PredicateBool::kFalse;
}

static arrow::Result<PredicateBool> EvalCompareArrayValue(const std::shared_ptr<arrow::Array>& array,
                                                          int64_t row,
                                                          const PredicateValue& value,
                                                          PredicateCompareOp op) {
  if (array->IsNull(row)) {
    return PredicateBool::kNull;
  }
  switch (array->type_id()) {
    case arrow::Type::INT32: {
      auto typed = std::static_pointer_cast<arrow::Int32Array>(array);
      return CompareIntegralValue(static_cast<int64_t>(typed->Value(row)), value, op);
    }
    case arrow::Type::INT64: {
      auto typed = std::static_pointer_cast<arrow::Int64Array>(array);
      return CompareIntegralValue(static_cast<int64_t>(typed->Value(row)), value, op);
    }
    case arrow::Type::FLOAT: {
      auto typed = std::static_pointer_cast<arrow::FloatArray>(array);
      return CompareFloatingValue(static_cast<double>(typed->Value(row)), value, op);
    }
    case arrow::Type::DOUBLE: {
      auto typed = std::static_pointer_cast<arrow::DoubleArray>(array);
      return CompareFloatingValue(typed->Value(row), value, op);
    }
    case arrow::Type::STRING: {
      auto typed = std::static_pointer_cast<arrow::StringArray>(array);
      return CompareStringValue(typed->GetString(row), value, op);
    }
    case arrow::Type::BOOL: {
      auto typed = std::static_pointer_cast<arrow::BooleanArray>(array);
      return CompareBoolValue(typed->Value(row), value, op);
    }
    default:
      return arrow::Status::Invalid("Unsupported predicate column type: ", array->type()->ToString());
  }
}

static arrow::Result<PredicateBool> EvalTermArrayValue(const std::shared_ptr<arrow::Array>& array,
                                                       int64_t row,
                                                       const std::vector<PredicateValue>& values) {
  if (array->IsNull(row)) {
    return PredicateBool::kNull;
  }
  for (const auto& value : values) {
    ARROW_ASSIGN_OR_RAISE(auto matched, EvalCompareArrayValue(array, row, value, PredicateCompareOp::kEqual));
    if (matched == PredicateBool::kTrue) {
      return PredicateBool::kTrue;
    }
  }
  return PredicateBool::kFalse;
}

static arrow::Result<PredicateBool> EvalPredicateNode(const std::shared_ptr<PredicateNode>& node,
                                                      const std::shared_ptr<arrow::RecordBatch>& batch,
                                                      int64_t row) {
  switch (node->kind) {
    case PredicateNodeKind::kCompare: {
      ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(batch->schema(), node->field_id));
      if (index < 0) {
        return arrow::Status::Invalid("Predicate field id ", node->field_id, " must be included in alive reader batch");
      }
      return EvalCompareArrayValue(batch->column(index), row, node->value, node->compare_op);
    }
    case PredicateNodeKind::kTerm: {
      ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(batch->schema(), node->field_id));
      if (index < 0) {
        return arrow::Status::Invalid("Predicate field id ", node->field_id, " must be included in alive reader batch");
      }
      return EvalTermArrayValue(batch->column(index), row, node->values);
    }
    case PredicateNodeKind::kAnd: {
      ARROW_ASSIGN_OR_RAISE(auto left, EvalPredicateNode(node->left, batch, row));
      ARROW_ASSIGN_OR_RAISE(auto right, EvalPredicateNode(node->right, batch, row));
      return PredicateAnd(left, right);
    }
    case PredicateNodeKind::kOr: {
      ARROW_ASSIGN_OR_RAISE(auto left, EvalPredicateNode(node->left, batch, row));
      ARROW_ASSIGN_OR_RAISE(auto right, EvalPredicateNode(node->right, batch, row));
      return PredicateOr(left, right);
    }
    case PredicateNodeKind::kNot: {
      ARROW_ASSIGN_OR_RAISE(auto child, EvalPredicateNode(node->child, batch, row));
      return PredicateNot(child);
    }
    case PredicateNodeKind::kIsNull:
    case PredicateNodeKind::kIsNotNull: {
      ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(batch->schema(), node->field_id));
      if (index < 0) {
        return arrow::Status::Invalid("Predicate field id ", node->field_id, " must be included in alive reader batch");
      }
      const bool is_null = batch->column(index)->IsNull(row);
      const bool result = node->kind == PredicateNodeKind::kIsNull ? is_null : !is_null;
      return result ? PredicateBool::kTrue : PredicateBool::kFalse;
    }
    case PredicateNodeKind::kAlwaysTrue:
      return PredicateBool::kTrue;
  }
  return arrow::Status::Invalid("Unknown predicate node kind");
}

// No reusable delete evaluator exists in milvus-storage yet: current delete
// readers live on the Milvus side and either materialize rows or parse legacy
// binlogs. This internal evaluator keeps storage-side alive stream semantics to
// manifest-derived delete evaluation only.
class DeleteEvaluator {
  public:
  static arrow::Result<std::shared_ptr<DeleteEvaluator>> Create(
      std::shared_ptr<Manifest> manifest,
      std::shared_ptr<arrow::Schema> schema,
      Properties properties,
      AliveReadOptions options,
      std::function<std::string(const std::string&)> key_retriever) {
    if (!manifest) {
      return arrow::Status::Invalid("DeleteEvaluator requires manifest");
    }
    auto evaluator = std::shared_ptr<DeleteEvaluator>(new DeleteEvaluator(std::move(manifest), std::move(schema),
                                                                          std::move(properties), std::move(options),
                                                                          std::move(key_retriever)));
    ARROW_RETURN_NOT_OK(evaluator->ValidateSupportedDeltaLogTypes());
    ARROW_RETURN_NOT_OK(evaluator->LoadPrimaryKeyDeletes());
    ARROW_RETURN_NOT_OK(evaluator->LoadPredicateDeletes());
    ARROW_RETURN_NOT_OK(evaluator->ResolveNeededColumns());
    return evaluator;
  }

  [[nodiscard]] bool empty() const {
    return int64_pk_delete_ts_.empty() && string_pk_delete_ts_.empty() && predicates_.empty();
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
    ARROW_ASSIGN_OR_RAISE(auto ts_index,
                          FindRequiredFieldIndex(batch->schema(), kMilvusTimestampFieldId, "row timestamp"));
    auto ts_array = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(ts_index));
    if (!ts_array) {
      return arrow::Status::Invalid("Row timestamp column must be int64");
    }

    if (HasPrimaryKeyDeletes()) {
      ARROW_ASSIGN_OR_RAISE(auto pk_index, FindRequiredFieldIndex(batch->schema(), *pk_field_id_, "primary key"));
      switch (batch->column(pk_index)->type_id()) {
        case arrow::Type::INT64:
          ARROW_RETURN_NOT_OK(EvaluateInt64PrimaryKey(batch, pk_index, *ts_array, mask->mutable_data()));
          break;
        case arrow::Type::STRING:
          ARROW_RETURN_NOT_OK(EvaluateStringPrimaryKey(batch, pk_index, *ts_array, mask->mutable_data()));
          break;
        default:
          return arrow::Status::Invalid(
              fmt::format("Unsupported primary-key column type: {}", batch->column(pk_index)->type()->ToString()));
      }
    }

    ARROW_RETURN_NOT_OK(EvaluatePredicateDeletes(batch, *ts_array, mask->mutable_data()));
    return std::make_shared<arrow::BooleanArray>(batch->num_rows(), std::move(mask), nullptr, 0);
  }

  private:
  struct CompiledPredicateDelete {
    uint64_t delete_ts;
    std::shared_ptr<PredicateNode> root;
    std::vector<int64_t> referenced_field_ids;
  };

  DeleteEvaluator(std::shared_ptr<Manifest> manifest,
                  std::shared_ptr<arrow::Schema> schema,
                  Properties properties,
                  AliveReadOptions options,
                  std::function<std::string(const std::string&)> key_retriever)
      : manifest_(std::move(manifest)),
        schema_(std::move(schema)),
        properties_(std::move(properties)),
        options_(std::move(options)),
        key_retriever_(std::move(key_retriever)) {}

  [[nodiscard]] bool HasPrimaryKeyDeletes() const { return !int64_pk_delete_ts_.empty() || !string_pk_delete_ts_.empty(); }

  arrow::Status ValidateSupportedDeltaLogTypes() const {
    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.type != DeltaLogType::PRIMARY_KEY && delta_log.type != DeltaLogType::PREDICATE) {
        return arrow::Status::Invalid("Unsupported delta log type for delete-aware masked reader: ",
                                      static_cast<int>(delta_log.type));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status LoadPrimaryKeyDeletes() {
    bool has_primary_key_delta_log = false;
    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.type == DeltaLogType::PRIMARY_KEY) {
        has_primary_key_delta_log = true;
      }
    }
    if (!has_primary_key_delta_log) {
      return arrow::Status::OK();
    }

    if (!schema_) {
      return arrow::Status::Invalid("PRIMARY_KEY delta logs require an explicit schema with PARQUET:field_id metadata");
    }

    ARROW_ASSIGN_OR_RAISE(auto pk_field_id, ResolvePrimaryKeyFieldId(*manifest_, schema_));
    if (!pk_field_id.has_value()) {
      return arrow::Status::Invalid(
          "PRIMARY_KEY delta logs require a resolvable primary-key field id. Add bloom_filter.<field_id> stats "
          "metadata to the manifest.");
    }
    pk_field_id_ = *pk_field_id;

    ARROW_ASSIGN_OR_RAISE(auto schema_pk_index, FindFieldIndexByFieldId(schema_, *pk_field_id_));
    if (schema_pk_index < 0) {
      return arrow::Status::Invalid(fmt::format("Primary-key field id {} not found in schema", *pk_field_id_));
    }
    ARROW_ASSIGN_OR_RAISE(auto schema_ts_index, FindFieldIndexByFieldId(schema_, kMilvusTimestampFieldId));
    if (schema_ts_index < 0) {
      return arrow::Status::Invalid("Row timestamp field id 1 not found in schema");
    }
    if (schema_->field(schema_ts_index)->type()->id() != arrow::Type::INT64) {
      return arrow::Status::Invalid("Row timestamp field id 1 must be int64");
    }

    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.type != DeltaLogType::PRIMARY_KEY) {
        continue;
      }
      if (delta_log.num_entries < 0) {
        return arrow::Status::Invalid("Delta log num_entries must be >= 0: ", delta_log.path);
      }
      if (delta_log.num_entries == 0) {
        continue;
      }
      ARROW_RETURN_NOT_OK(LoadPrimaryKeyDeltaLog(delta_log));
    }
    return arrow::Status::OK();
  }

  arrow::Status LoadPrimaryKeyDeltaLog(const DeltaLog& delta_log) {
    ARROW_ASSIGN_OR_RAISE(auto reader, OpenDeltaLogReader(delta_log, properties_, key_retriever_));
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      if (batch->num_columns() < 2) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log must contain pk and ts columns: ", delta_log.path);
      }
      std::shared_ptr<arrow::Array> pk_array = batch->GetColumnByName(kDeltaPkColumnName);
      std::shared_ptr<arrow::Array> ts_column = batch->GetColumnByName(kDeltaTsColumnName);
      if (!pk_array || !ts_column) {
        // StorageV2 deltalogs written through Milvus packed writer may use
        // field-id column names ("0", "1") while preserving the logical order:
        // pk first, delete timestamp second.
        pk_array = batch->column(0);
        ts_column = batch->column(1);
      }
      auto ts_array = std::dynamic_pointer_cast<arrow::Int64Array>(ts_column);
      if (!ts_array) {
        return arrow::Status::Invalid("PRIMARY_KEY delta log column 'ts' must be int64: ", delta_log.path);
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


  arrow::Status LoadPredicateDeletes() {
    bool has_predicate_delta_log = false;
    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.type == DeltaLogType::PREDICATE) {
        has_predicate_delta_log = true;
        break;
      }
    }
    if (!has_predicate_delta_log) {
      return arrow::Status::OK();
    }
    if (!schema_) {
      return arrow::Status::Invalid("PREDICATE delta logs require an explicit schema with PARQUET:field_id metadata");
    }
    ARROW_ASSIGN_OR_RAISE(auto schema_ts_index, FindFieldIndexByFieldId(schema_, kMilvusTimestampFieldId));
    if (schema_ts_index < 0) {
      return arrow::Status::Invalid("Row timestamp field id 1 not found in schema");
    }
    if (schema_->field(schema_ts_index)->type()->id() != arrow::Type::INT64) {
      return arrow::Status::Invalid("Row timestamp field id 1 must be int64");
    }

    for (const auto& delta_log : manifest_->deltaLogs()) {
      if (delta_log.type != DeltaLogType::PREDICATE) {
        continue;
      }
      if (delta_log.num_entries < 0) {
        return arrow::Status::Invalid("Delta log num_entries must be >= 0: ", delta_log.path);
      }
      if (delta_log.num_entries == 0) {
        continue;
      }
      ARROW_RETURN_NOT_OK(LoadPredicateDeltaLog(delta_log));
    }
    return arrow::Status::OK();
  }

  arrow::Status LoadPredicateDeltaLog(const DeltaLog& delta_log) {
    ARROW_ASSIGN_OR_RAISE(auto reader, OpenDeltaLogReader(delta_log, properties_, key_retriever_));
    ARROW_RETURN_NOT_OK(ValidatePredicateDeltaSchema(reader->schema(), delta_log.path));
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_RETURN_NOT_OK(reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(ConsumePredicateDeleteBatch(batch, delta_log.path));
    }
    return reader->Close();
  }

  arrow::Status ValidatePredicateDeltaSchema(const std::shared_ptr<arrow::Schema>& schema, const std::string& path) const {
    if (!schema || !schema->metadata()) {
      return arrow::Status::Invalid("PREDICATE delta log requires milvus.predicate_delta.format_version=1: ", path);
    }
    auto version = schema->metadata()->Get(kPredicateDeltaFormatVersionKey);
    if (!version.ok() || version.ValueOrDie() != "1") {
      return arrow::Status::Invalid("PREDICATE delta log requires milvus.predicate_delta.format_version=1: ", path);
    }
    auto delete_ts_field = schema->GetFieldByName(kPredicateDeltaDeleteTsColumnName);
    auto plan_field = schema->GetFieldByName(kPredicateDeltaSerializedPlanColumnName);
    if (!delete_ts_field || !plan_field) {
      return arrow::Status::Invalid("PREDICATE delta log requires delete_timestamp and serialized_expr_plan columns: ", path);
    }
    if (delete_ts_field->type()->id() != arrow::Type::INT64 || plan_field->type()->id() != arrow::Type::BINARY) {
      return arrow::Status::Invalid("PREDICATE delta log delete_timestamp must be int64 and serialized_expr_plan must be binary: ",
                                    path);
    }
    return arrow::Status::OK();
  }

  arrow::Status ConsumePredicateDeleteBatch(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& path) {
    auto delete_ts_column = batch->GetColumnByName(kPredicateDeltaDeleteTsColumnName);
    auto plan_column = batch->GetColumnByName(kPredicateDeltaSerializedPlanColumnName);
    auto delete_ts_array = std::dynamic_pointer_cast<arrow::Int64Array>(delete_ts_column);
    auto plan_array = std::dynamic_pointer_cast<arrow::BinaryArray>(plan_column);
    if (!delete_ts_array || !plan_array) {
      return arrow::Status::Invalid("PREDICATE delta log delete_timestamp must be int64 and serialized_expr_plan must be binary: ",
                                    path);
    }

    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (delete_ts_array->IsNull(i) || plan_array->IsNull(i)) {
        return arrow::Status::Invalid("PREDICATE delta log delete_timestamp/serialized_expr_plan must not contain nulls: ", path);
      }
      ARROW_ASSIGN_OR_RAISE(auto delete_ts, ValidateDeleteTimestamp(delete_ts_array->Value(i), path));
      if (options_.visible_until_ts.has_value() && delete_ts > *options_.visible_until_ts) {
        continue;
      }

      int32_t plan_length = 0;
      const uint8_t* plan_data = plan_array->GetValue(i, &plan_length);
      ARROW_ASSIGN_OR_RAISE(auto parsed_plan, ParsePredicatePlan(std::string_view(reinterpret_cast<const char*>(plan_data),
                                                                                   static_cast<size_t>(plan_length))));

      predicates_.push_back(CompiledPredicateDelete{delete_ts, std::move(parsed_plan.root),
                                                    std::move(parsed_plan.referenced_field_ids)});
    }
    return arrow::Status::OK();
  }

  arrow::Status ResolveNeededColumns() {
    needed_columns_.clear();
    std::unordered_set<std::string> seen;
    auto add_field_id = [&](int64_t field_id, const std::string& role) -> arrow::Status {
      ARROW_ASSIGN_OR_RAISE(auto index, FindFieldIndexByFieldId(schema_, field_id));
      if (index < 0) {
        return arrow::Status::Invalid(fmt::format("{} field id {} not found in schema", role, field_id));
      }
      const auto& name = schema_->field(index)->name();
      if (seen.insert(name).second) {
        needed_columns_.push_back(name);
      }
      return arrow::Status::OK();
    };

    if (HasPrimaryKeyDeletes()) {
      if (!pk_field_id_.has_value()) {
        return arrow::Status::Invalid("PRIMARY_KEY delta logs require primary-key field id");
      }
      ARROW_RETURN_NOT_OK(add_field_id(*pk_field_id_, "Primary-key"));
    }
    if (HasPrimaryKeyDeletes() || !predicates_.empty()) {
      ARROW_RETURN_NOT_OK(add_field_id(kMilvusTimestampFieldId, "Row timestamp"));
    }
    for (const auto& predicate : predicates_) {
      for (const auto field_id : predicate.referenced_field_ids) {
        ARROW_RETURN_NOT_OK(add_field_id(field_id, "Predicate"));
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
          fmt::format("{} field id {} must be included in alive reader output schema", role, field_id));
    }
    return index;
  }


  arrow::Status EvaluatePredicateDeletes(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         const arrow::Int64Array& ts_array,
                                         uint8_t* mask) const {
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (ts_array.IsNull(i) || !arrow::bit_util::GetBit(mask, i)) {
        continue;
      }
      const auto row_ts = static_cast<uint64_t>(ts_array.Value(i));
      for (const auto& predicate : predicates_) {
        if (row_ts > predicate.delete_ts) {
          continue;
        }
        ARROW_ASSIGN_OR_RAISE(auto value, EvalPredicateNode(predicate.root, batch, i));
        if (value == PredicateBool::kTrue) {
          arrow::bit_util::ClearBit(mask, i);
          break;
        }
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status EvaluateInt64PrimaryKey(const std::shared_ptr<arrow::RecordBatch>& batch,
                                        int pk_index,
                                        const arrow::Int64Array& ts_array,
                                        uint8_t* mask) const {
    auto pk_array = std::static_pointer_cast<arrow::Int64Array>(batch->column(pk_index));
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (pk_array->IsNull(i) || ts_array.IsNull(i)) {
        continue;
      }
      auto it = int64_pk_delete_ts_.find(pk_array->Value(i));
      if (it != int64_pk_delete_ts_.end() && static_cast<uint64_t>(ts_array.Value(i)) <= it->second) {
        arrow::bit_util::ClearBit(mask, i);
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status EvaluateStringPrimaryKey(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         int pk_index,
                                         const arrow::Int64Array& ts_array,
                                         uint8_t* mask) const {
    auto pk_array = std::static_pointer_cast<arrow::StringArray>(batch->column(pk_index));
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      if (pk_array->IsNull(i) || ts_array.IsNull(i)) {
        continue;
      }
      auto it = string_pk_delete_ts_.find(pk_array->GetString(i));
      if (it != string_pk_delete_ts_.end() && static_cast<uint64_t>(ts_array.Value(i)) <= it->second) {
        arrow::bit_util::ClearBit(mask, i);
      }
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<Manifest> manifest_;
  std::shared_ptr<arrow::Schema> schema_;
  Properties properties_;
  AliveReadOptions options_;
  std::function<std::string(const std::string&)> key_retriever_;
  std::optional<int64_t> pk_field_id_;
  std::unordered_map<int64_t, uint64_t> int64_pk_delete_ts_;
  std::unordered_map<std::string, uint64_t> string_pk_delete_ts_;
  std::vector<CompiledPredicateDelete> predicates_;
  std::vector<std::string> needed_columns_;
};


arrow::Result<std::shared_ptr<DeleteEvaluator>> CreateDeleteEvaluator(
    std::shared_ptr<Manifest> manifest,
    std::shared_ptr<arrow::Schema> schema,
    Properties properties,
    AliveReadOptions options,
    std::function<std::string(const std::string&)> key_retriever) {
  return DeleteEvaluator::Create(std::move(manifest), std::move(schema), std::move(properties), std::move(options),
                                 std::move(key_retriever));
}

const std::vector<std::string>& DeleteEvaluatorNeededColumns(const std::shared_ptr<DeleteEvaluator>& evaluator) {
  return evaluator->NeededColumns();
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> EvaluateDeleteKeepMask(
    const std::shared_ptr<DeleteEvaluator>& evaluator,
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  return evaluator->EvaluateKeepMask(batch);
}

}  // namespace api
}  // namespace milvus_storage
