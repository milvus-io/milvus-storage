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

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/localfs.h>
#include <gtest/gtest.h>
#include <arrow/api.h>
#include <packed/writer.h>
#include <parquet/properties.h>
#include <packed/reader.h>
#include <memory>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include "test_util.h"
#include "filesystem/fs.h"
#include "packed_test_base.h"

namespace milvus_storage {

class PackedIntegrationTest : public PackedTestBase {
  protected:
  PackedIntegrationTest() : props_(*parquet::default_writer_properties()) {}

  void SetUp() override {
    const char* access_key = std::getenv("ACCESS_KEY");
    const char* secret_key = std::getenv("SECRET_KEY");
    const char* endpoint_url = std::getenv("S3_ENDPOINT_URL");
    const char* file_path = std::getenv("FILE_PATH");
    std::string uri = "file:///tmp/";
    if (access_key != nullptr && secret_key != nullptr && endpoint_url != nullptr && file_path != nullptr) {
      uri = endpoint_url;
    }
    auto factory = std::make_shared<FileSystemFactory>();
    ASSERT_AND_ASSIGN(fs_, factory->BuildFileSystem(uri, &file_path_));

    SetUpCommonData();
    props_ = *parquet::default_writer_properties();
    writer_memory_ = (22 + 16) * 1024 * 1024;  // 22 MB memory is for s3fs part upload
    reader_memory_ = 16 * 1024 * 1024;         // 16 MB memory for reading
  }

  void TearDown() override { fs_->DeleteDir(file_path_); }

  size_t writer_memory_;
  size_t reader_memory_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string file_path_;
  parquet::WriterProperties props_;
  const int bath_size = 100000;
};

TEST_F(PackedIntegrationTest, WriteAndRead) {
  PackedRecordBatchWriter writer(writer_memory_, schema_, *fs_, file_path_, props_);
  for (int i = 0; i < bath_size; ++i) {
    EXPECT_TRUE(writer.Write(record_batch_).ok());
  }
  EXPECT_TRUE(writer.Close().ok());

  std::set<int> needed_columns = {0, 1, 2};
  std::vector<ColumnOffset> column_offsets = {
      ColumnOffset(0, 0),
      ColumnOffset(1, 0),
      ColumnOffset(1, 1),
  };

  auto paths = std::vector<std::string>{file_path_ + "/0", file_path_ + "/1"};

  // after writing, the column of large_str is in 0th file, and the last int64 columns are in 1st file
  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("str", arrow::utf8()),
      arrow::field("int32", arrow::int32()),
      arrow::field("int64", arrow::int64()),
  };
  auto new_schema = arrow::schema(fields);

  PackedRecordBatchReader pr(*fs_, paths, new_schema, column_offsets, needed_columns, reader_memory_);
  ASSERT_AND_ARROW_ASSIGN(auto table, pr.ToTable());
  ASSERT_STATUS_OK(pr.Close());

  int64_t total_rows = table->num_rows();

  auto chunks = table->GetColumnByName("str");
  int64_t count = 0;
  for (int i = 0; i < chunks->num_chunks(); ++i) {
    auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(chunks->chunk(i));
    for (int j = 0; j < str_array->length(); ++j) {
      int expected_index = (count++) % str_values.size();
      ASSERT_EQ(str_array->GetString(j), str_values[expected_index]);
    }
  }

  chunks = table->GetColumnByName("int32");
  count = 0;
  for (int i = 0; i < chunks->num_chunks(); ++i) {
    auto int32_array = std::dynamic_pointer_cast<arrow::Int32Array>(chunks->chunk(i));
    for (int j = 0; j < int32_array->length(); ++j) {
      int expected_index = (count++) % int32_values.size();
      ASSERT_EQ(int32_array->Value(j), int32_values[expected_index]);
    }
  }

  chunks = table->GetColumnByName("int64");
  count = 0;
  for (int i = 0; i < chunks->num_chunks(); ++i) {
    auto int64_array = std::dynamic_pointer_cast<arrow::Int64Array>(chunks->chunk(i));
    for (int j = 0; j < int64_array->length(); ++j) {
      int expected_index = (count++) % int64_values.size();
      ASSERT_EQ(int64_array->Value(j), int64_values[expected_index]);
    }
  }
}

}  // namespace milvus_storage