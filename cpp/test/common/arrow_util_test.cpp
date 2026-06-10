// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <arrow/api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "milvus-storage/common/arrow_util.h"
#include "test_env.h"

namespace milvus_storage::test {
namespace {

arrow::Result<std::shared_ptr<arrow::Array>> BuildInt64Array(const std::vector<int64_t>& values) {
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(values));
  return builder.Finish();
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildStringViewArray(const std::vector<std::string>& values) {
  arrow::StringViewBuilder builder;
  for (const auto& value : values) {
    ARROW_RETURN_NOT_OK(builder.Append(value));
  }
  return builder.Finish();
}

std::unordered_set<const void*> VariadicBufferAddresses(const std::shared_ptr<arrow::ChunkedArray>& column) {
  std::unordered_set<const void*> addresses;
  for (const auto& chunk : column->chunks()) {
    const auto& buffers = chunk->data()->buffers;
    for (size_t i = 2; i < buffers.size(); ++i) {
      if (buffers[i] && buffers[i]->data()) {
        addresses.insert(buffers[i]->data());
      }
    }
  }
  return addresses;
}

}  // namespace

TEST(ArrowUtilTest, CopySelectedRowsMaterializesAcrossChunks) {
  ASSERT_AND_ASSIGN(auto ids0, BuildInt64Array({10, 20}));
  ASSERT_AND_ASSIGN(auto ids1, BuildInt64Array({30, 40}));
  ASSERT_AND_ASSIGN(auto strings0,
                    BuildStringViewArray({"zero-value-longer-than-inline", "one-value-longer-than-inline"}));
  ASSERT_AND_ASSIGN(auto strings1,
                    BuildStringViewArray({"two-value-longer-than-inline", "three-value-longer-than-inline"}));
  auto id_column = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{ids0, ids1});
  auto string_column = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{strings0, strings1});
  auto table =
      arrow::Table::Make(arrow::schema({arrow::field("id", arrow::int64()), arrow::field("text", arrow::utf8_view())}),
                         {id_column, string_column});

  const auto input_string_buffers = VariadicBufferAddresses(string_column);
  ASSERT_FALSE(input_string_buffers.empty());

  ASSERT_AND_ASSIGN(auto copied, CopySelectedRows(table, {3, 0, 2}));

  ASSERT_EQ(copied->num_rows(), 3);
  ASSERT_EQ(copied->column(0)->num_chunks(), 1);
  ASSERT_EQ(copied->column(1)->num_chunks(), 1);
  ASSERT_EQ(copied->column(1)->type()->id(), arrow::Type::STRING_VIEW);

  auto copied_ids = std::static_pointer_cast<arrow::Int64Array>(copied->column(0)->chunk(0));
  EXPECT_EQ(copied_ids->Value(0), 40);
  EXPECT_EQ(copied_ids->Value(1), 10);
  EXPECT_EQ(copied_ids->Value(2), 30);

  auto copied_strings = std::static_pointer_cast<arrow::StringViewArray>(copied->column(1)->chunk(0));
  EXPECT_EQ(copied_strings->GetView(0), "three-value-longer-than-inline");
  EXPECT_EQ(copied_strings->GetView(1), "zero-value-longer-than-inline");
  EXPECT_EQ(copied_strings->GetView(2), "two-value-longer-than-inline");

  const auto output_string_buffers = VariadicBufferAddresses(copied->column(1));
  ASSERT_FALSE(output_string_buffers.empty());
  for (const auto* address : output_string_buffers) {
    EXPECT_EQ(input_string_buffers.count(address), 0);
  }
}

TEST(ArrowUtilTest, CopySelectedRowsRejectsInvalidIndices) {
  ASSERT_AND_ASSIGN(auto values, BuildInt64Array({10, 20}));
  auto table = arrow::Table::Make(arrow::schema({arrow::field("id", arrow::int64())}), {values});

  EXPECT_FALSE(CopySelectedRows(table, {-1}).ok());
  EXPECT_FALSE(CopySelectedRows(table, {2}).ok());
}

}  // namespace milvus_storage::test
