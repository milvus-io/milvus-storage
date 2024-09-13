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

#include "packed_test_base.h"

namespace milvus_storage {

class OneFileTest : public PackedTestBase {};

TEST_F(OneFileTest, WriteAndRead) {
  int batch_size = 100;

  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(0, 1),
      ColumnOffset(0, 2),
  };

  std::vector<std::string> paths = {file_path_ + "/0"};

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
      arrow::field("str", arrow::utf8()),
  };

  TestWriteAndRead(batch_size, paths, fields, column_offsets);
}

}  // namespace milvus_storage