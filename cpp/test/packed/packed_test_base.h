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
#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include "arrow/table.h"

#include "test_util.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/constants.h"
#include <milvus-storage/packed/writer.h>
#include <milvus-storage/packed/reader.h>
#include <milvus-storage/packed/column_group.h>
#include <memory>
#include <gtest/gtest.h>
#include <parquet/properties.h>
#include <vector>
#include <string>
#include "boost/filesystem/path.hpp"
#include <boost/filesystem/operations.hpp>

namespace milvus_storage {
// Environment variables to configure the S3 test environment
static const char* kEnvAccessKey = "ACCESS_KEY";
static const char* kEnvSecretKey = "SECRET_KEY";
static const char* kEnvAddress = "ADDRESS";
static const char* kEnvCloudProvider = "CLOUD_PROVIDER";
static const char* kEnvBucketName = "BUCKET_NAME";
static const char* kEnvRegion = "REGION";

class PackedTestBase : public ::testing::Test {
  protected:
  void SetUp() override {
    const char* access_key = std::getenv(kEnvAccessKey);
    const char* secret_key = std::getenv(kEnvSecretKey);
    const char* address = std::getenv(kEnvAddress);
    const char* cloud_provider = std::getenv(kEnvCloudProvider);
    const char* bucket_name = std::getenv(kEnvBucketName);
    const char* region = std::getenv(kEnvRegion);

    storage_config_ = StorageConfig();

    path_ = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    auto conf = ArrowFileSystemConfig();
    conf.storage_type = "local";
    conf.root_path = path_.string();

    if (cloud_provider != nullptr) {
      storage_config_.part_size = 10 * 1024 * 1024;  // 10 MB for S3FS part upload
      conf.cloud_provider = std::string(cloud_provider);
      conf.root_path = boost::filesystem::unique_path().string();
      path_ = conf.root_path;
      conf.storage_type = "remote";
      conf.requestTimeoutMs = 10000;
      conf.useSSL = true;
      conf.log_level = "debug";
      conf.region = std::string(region);
      conf.address = std::string(address);
      conf.bucket_name = std::string(bucket_name);
      conf.useVirtualHost = false;
      // not use iam
      if (access_key != nullptr && secret_key != nullptr) {
        conf.useIAM = false;
        conf.access_key_id = std::string(access_key);
        conf.access_key_value = std::string(secret_key);
      } else {
        conf.useIAM = true;
        conf.access_key_id = "";
        conf.access_key_value = "";
        // azure should provide access key
        if (conf.cloud_provider == "azure") {
          conf.access_key_id = std::string(access_key);
        }
      }
    }

    ArrowFileSystemSingleton::GetInstance().Init(conf);
    fs_ = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();

    SetUpCommonData();
    writer_memory_ = (22 + 16) * 1024 * 1024;  // 22 MB for S3FS part upload
    reader_memory_ = 5 * 1024 * 1024;
  }

  void TearDown() override {
    boost::filesystem::remove_all(path_);
    ArrowFileSystemSingleton::GetInstance().Release();
  }

  void ValidateTableData(const std::shared_ptr<arrow::Table>& table) {
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

  void SetupOneFile() {
    one_file_path_ = path_.string() + "/10000.parquet";
    std::vector<std::string> paths = {one_file_path_};
    int batch_size = 100;
    auto column_groups = std::vector<std::vector<int>>{{0, 1, 2}};
    PackedRecordBatchWriter writer(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);
    for (int i = 0; i < batch_size; ++i) {
      EXPECT_TRUE(writer.Write(record_batch_).ok());
    }
    auto column_index_groups = writer.Close();
  }

  void SetUpCommonData() {
    record_batch_ = randomRecordBatch();
    table_ = arrow::Table::FromRecordBatches({record_batch_}).ValueOrDie();
    schema_ = table_->schema();
  }

  protected:
  std::shared_ptr<arrow::RecordBatch> randomRecordBatch() {
    arrow::Int32Builder int_builder;
    arrow::Int64Builder int64_builder;
    arrow::StringBuilder str_builder;

    int32_values = {rand() % 10000, rand() % 10000, rand() % 10000};
    int64_values = {rand() % 10000000, rand() % 10000000, rand() % 10000000};
    str_values = {random_string(10000), random_string(10000), random_string(10000)};

    int_builder.AppendValues(int32_values).ok();
    int64_builder.AppendValues(int64_values).ok();
    str_builder.AppendValues(str_values).ok();

    std::shared_ptr<arrow::Array> int_array;
    std::shared_ptr<arrow::Array> int64_array;
    std::shared_ptr<arrow::Array> str_array;

    int_builder.Finish(&int_array).ok();
    int64_builder.Finish(&int64_array).ok();
    str_builder.Finish(&str_array).ok();

    std::vector<std::shared_ptr<arrow::Array>> arrays = {int_array, int64_array, str_array};
    auto schema = arrow::schema(
        {arrow::field("int32", arrow::int32(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"})),
         arrow::field("int64", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"200"})),
         arrow::field("str", arrow::utf8(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"300"}))});
    return arrow::RecordBatch::Make(schema, 3, arrays);
  }

  std::string random_string(size_t length) {
    auto randchar = []() -> char {
      const char charset[] =
          "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
  }

  size_t writer_memory_;
  size_t reader_memory_;
  ArrowFileSystemPtr fs_;
  boost::filesystem::path path_;

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::shared_ptr<arrow::Table> table_;

  std::vector<int32_t> int32_values;
  std::vector<int64_t> int64_values;
  std::vector<std::basic_string<char>> str_values;

  StorageConfig storage_config_;
  std::string one_file_path_;
};

}  // namespace milvus_storage