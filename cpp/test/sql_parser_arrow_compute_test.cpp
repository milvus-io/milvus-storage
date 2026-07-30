// Copyright 2026 Zilliz
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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/compute/exec.h>
#include <arrow/compute/expression.h>

#include "common/sql_predicate_arrow.h"
#include "test_env.h"

namespace milvus_storage::test {
namespace {

namespace cp = arrow::compute;

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakePredicateInputBatch() {
  arrow::Int64Builder a_builder;
  ARROW_RETURN_NOT_OK(a_builder.AppendValues({10, 16, 15, 30}));
  std::shared_ptr<arrow::Array> a;
  ARROW_RETURN_NOT_OK(a_builder.Finish(&a));

  arrow::Int64Builder b_builder;
  ARROW_RETURN_NOT_OK(b_builder.AppendValues({12, 12, 13, 12}));
  std::shared_ptr<arrow::Array> b;
  ARROW_RETURN_NOT_OK(b_builder.Finish(&b));

  arrow::StringBuilder s_builder;
  ARROW_RETURN_NOT_OK(s_builder.AppendValues({"x", "y", "x", "z"}));
  std::shared_ptr<arrow::Array> s;
  ARROW_RETURN_NOT_OK(s_builder.Finish(&s));

  arrow::DoubleBuilder f_builder;
  ARROW_RETURN_NOT_OK(f_builder.AppendValues({1.0, 2.5, 3.5, 4.0}));
  std::shared_ptr<arrow::Array> f;
  ARROW_RETURN_NOT_OK(f_builder.Finish(&f));

  arrow::BooleanBuilder flag_builder;
  ARROW_RETURN_NOT_OK(flag_builder.AppendValues(std::vector<bool>{true, false, true, false}));
  std::shared_ptr<arrow::Array> flag;
  ARROW_RETURN_NOT_OK(flag_builder.Finish(&flag));

  arrow::Int64Builder nullable_builder;
  ARROW_RETURN_NOT_OK(nullable_builder.AppendNull());
  ARROW_RETURN_NOT_OK(nullable_builder.Append(5));
  ARROW_RETURN_NOT_OK(nullable_builder.AppendNull());
  ARROW_RETURN_NOT_OK(nullable_builder.Append(7));
  std::shared_ptr<arrow::Array> nullable;
  ARROW_RETURN_NOT_OK(nullable_builder.Finish(&nullable));

  auto schema =
      arrow::schema({arrow::field("a", arrow::int64(), false), arrow::field("b", arrow::int64(), false),
                     arrow::field("s", arrow::utf8(), false), arrow::field("f", arrow::float64(), false),
                     arrow::field("flag", arrow::boolean(), false), arrow::field("nullable", arrow::int64(), true)});
  return arrow::RecordBatch::Make(schema, a->length(), {a, b, s, f, flag, nullable});
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> ExecutePredicate(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                                     const std::string& predicate) {
  ARROW_ASSIGN_OR_RAISE(auto expr, ParseSqlPredicateToArrowExpression(predicate, batch->schema()));
  ARROW_ASSIGN_OR_RAISE(auto bound_expr, expr.Bind(*batch->schema()));
  ARROW_ASSIGN_OR_RAISE(auto exec_batch, cp::MakeExecBatch(*batch->schema(), arrow::Datum(batch)));
  ARROW_ASSIGN_OR_RAISE(auto result_datum, cp::ExecuteScalarExpression(bound_expr, exec_batch));
  auto result_array = result_datum.make_array();
  if (result_array->type_id() != arrow::Type::BOOL) {
    return arrow::Status::Invalid("Predicate result must be boolean, got ", result_array->type()->ToString());
  }
  return std::static_pointer_cast<arrow::BooleanArray>(result_array);
}

void ExpectBooleanValues(const arrow::BooleanArray& array, const std::vector<bool>& expected) {
  ASSERT_EQ(array.length(), static_cast<int64_t>(expected.size()));
  for (int64_t i = 0; i < array.length(); ++i) {
    ASSERT_FALSE(array.IsNull(i)) << "row " << i;
    EXPECT_EQ(array.Value(i), expected[i]) << "row " << i;
  }
}

void ExpectPredicate(const std::shared_ptr<arrow::RecordBatch>& batch,
                     const std::string& predicate,
                     const std::vector<bool>& expected) {
  ASSERT_AND_ASSIGN(auto result, ExecutePredicate(batch, predicate));
  ExpectBooleanValues(*result, expected);
}

void ExpectRejects(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& predicate) {
  EXPECT_FALSE(ParseSqlPredicateToArrowExpression(predicate, batch->schema()).ok()) << predicate;
}

}  // namespace

TEST(SqlParserArrowComputeTest, ParsesGreaterThanPredicateAndExecutesOnRecordBatch) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectPredicate(batch, "a > 15", {false, true, false, true});
}

TEST(PredicateSqlToArrowTest, SupportsComparisonsAndLogicalOperators) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectPredicate(batch, "a > 15 and (b = 12 or s = 'x')", {false, true, false, true});
  ExpectPredicate(batch, "not (a <= 15)", {false, true, false, true});
  ExpectPredicate(batch, "a != 15 and b >= 12 and b < 13", {true, true, false, true});
}

TEST(PredicateSqlToArrowTest, SupportsStringFloatBoolAndInLiterals) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectPredicate(batch, "s in ('x', 'z')", {true, false, true, true});
  ExpectPredicate(batch, "a in (10, 30)", {true, false, false, true});
  ExpectPredicate(batch, "f >= 2.5 and flag = true", {false, false, true, false});
}

TEST(PredicateSqlToArrowTest, SupportsNegativeNumericLiterals) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectPredicate(batch, "a > -5", {true, true, true, true});
  ExpectPredicate(batch, "a in (10, -5)", {true, false, false, false});
  ExpectPredicate(batch, "f >= -2.5", {true, true, true, true});
  // A '-' not followed by a digit (unary minus) is still rejected.
  ExpectRejects(batch, "-a > 1");
}

TEST(PredicateSqlToArrowTest, SupportsIsNullAndIsNotNull) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectPredicate(batch, "nullable is null", {true, false, true, false});
  ExpectPredicate(batch, "nullable is not null", {false, true, false, true});
}

TEST(PredicateSqlToArrowTest, RejectsUnknownAndQualifiedColumns) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectRejects(batch, "missing = 1");
  ExpectRejects(batch, "t.a = 1");
}

TEST(PredicateSqlToArrowTest, RejectsUnsupportedSyntax) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  const std::vector<std::string> unsupported = {
      "abs(a) = 1",
      "cast(a as int) = 1",
      "s like 'x%'",
      "a + 1 > 2",
      "a in (select a from t)",
      "a in (1, null)",
      "a = null",
      "a between 1 and 3",
      "$meta = 1",
      "? = 1",
      "date '2024-01-01' = date '2024-01-01'",
      "extract(year from f) = 2024",
      "-a > 1",
      "case when flag then a else b end = 10",
  };
  for (const auto& predicate : unsupported) {
    ExpectRejects(batch, predicate);
  }
}

TEST(PredicateSqlToArrowTest, RejectsMalformedAndEmptyPredicate) {
  ASSERT_AND_ASSIGN(auto batch, MakePredicateInputBatch());
  ExpectRejects(batch, "");
  ExpectRejects(batch, "a >");
}

}  // namespace milvus_storage::test
