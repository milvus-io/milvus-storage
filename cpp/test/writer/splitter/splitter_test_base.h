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

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <gtest/gtest.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <memory>
#include "test_util.h"

using namespace std;

namespace milvus_storage {

class SplitterTestBase : public ::testing::Test {
  protected:
  void SetUpCommonData() {
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    ASSERT_STATUS_OK(int_builder.AppendValues({1, 2, 3}));
    ASSERT_STATUS_OK(int64_builder.AppendValues({1, 2, 3}));
    ASSERT_STATUS_OK(
        str_builder.AppendValues({std::string(10000, 'a'), std::string(10000, 'b'), std::string(10000, 'c')}));

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> int64_array;
    std::shared_ptr<arrow::Array> str_array;

    ASSERT_STATUS_OK(int_builder.Finish(&int_array));
    ASSERT_STATUS_OK(int64_builder.Finish(&int64_array));
    ASSERT_STATUS_OK(str_builder.Finish(&str_array));

    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field("int", arrow::int32()), arrow::field("int64", arrow::int64()), arrow::field("str", arrow::utf8())};
    std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};

    record_batch_ = arrow::RecordBatch::Make(arrow::schema(fields), 3, arrays);
  }

  std::shared_ptr<arrow::RecordBatch> record_batch_;
};

}  // namespace milvus_storage