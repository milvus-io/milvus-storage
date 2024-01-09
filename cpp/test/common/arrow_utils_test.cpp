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

#include <arrow/type_fwd.h>
#include <boost/filesystem/operations.hpp>
#include "common/arrow_util.h"
#include "common/fs_util.h"
#include "test_util.h"
#include "gtest/gtest.h"
#include "boost/filesystem/path.hpp"

namespace milvus_storage {

class ArrowUtilsTest : public testing::Test {
  protected:
  void SetUp() override {
    path_ = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    boost::filesystem::create_directories(path_);
  }
  void TearDown() override { boost::filesystem::remove_all(path_); }
  boost::filesystem::path path_;
};

TEST_F(ArrowUtilsTest, TestMakeArrowRecordBatchReader) {
  std::string out;
  ASSERT_AND_ASSIGN(auto fs, BuildFileSystem("file://" + path_.string(), &out));
  auto file_path = path_.string() + "/test.parquet";
  auto schema = CreateArrowSchema({"f_int64"}, {arrow::int64()});
  ASSERT_STATUS_OK(PrepareSimpleParquetFile(schema, *fs, file_path, 1));
  ASSERT_AND_ASSIGN(auto file_reader, MakeArrowFileReader(*fs, file_path));
  ASSERT_AND_ASSIGN(auto batch_reader, MakeArrowRecordBatchReader(*file_reader, schema, {.primary_column = "f_int64"}));
  ASSERT_AND_ARROW_ASSIGN(auto batch, batch_reader->Next());
  ASSERT_EQ(1, batch->num_rows());
}
}  // namespace milvus_storage
