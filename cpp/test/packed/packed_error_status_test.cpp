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

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <parquet/arrow/writer.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/packed/writer.h"

#include "packed_test_base.h"

namespace milvus_storage {

namespace {

void ExpectPackedCode(const arrow::Status& status, ExtendStatusCode code) {
  ASSERT_FALSE(status.ok());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), code);
}

void ExpectExceptionMessageContainsCode(const std::function<void()>& fn, const std::string& code_name) {
  try {
    fn();
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(code_name), std::string::npos) << e.what();
  }
}

}  // namespace

class PackedErrorStatusTest : public PackedTestBase {};

TEST_F(PackedErrorStatusTest, WriterPathGroupMismatchIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{0}, {1}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterColumnIndexOutOfRangeIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{schema_->num_fields()}};

  auto result = PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_);

  ExpectPackedCode(result.status(), ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, WriterRecordBatchColumnMismatchIsInvalidArgs) {
  std::vector<std::string> paths = {path_ + "/0.parquet"};
  std::vector<std::vector<int>> column_groups = {{0, 1, 2}};
  ASSERT_AND_ASSIGN(auto writer,
                    PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config_, column_groups, writer_memory_));
  ASSERT_AND_ASSIGN(auto short_batch, record_batch_->SelectColumns({0, 1}));

  auto status = writer->Write(short_batch);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
  (void)writer->Close();
}

TEST_F(PackedErrorStatusTest, ColumnGroupNullBatchIsInvalidArgs) {
  ColumnGroup group(0, {0});

  auto status = group.AddRecordBatch(nullptr);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
}

TEST_F(PackedErrorStatusTest, ReaderMissingFileIsStorageIO) {
  std::vector<std::string> paths = {path_ + "/missing.parquet"};

  ExpectExceptionMessageContainsCode([&]() { PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_); },
                                     "PackedStorageIO");
}

TEST_F(PackedErrorStatusTest, ReaderNullOutputPointerIsInvalidArgs) {
  SetupOneFile();
  std::vector<std::string> paths = {one_file_path_};
  PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_);

  auto status = reader.ReadNext(nullptr);

  ExpectPackedCode(status, ExtendStatusCode::PackedInvalidArgs);
  ASSERT_STATUS_OK(reader.Close());
}

TEST_F(PackedErrorStatusTest, ReaderMissingPackedMetadataIsMetadataCorrupted) {
  auto parquet_path = path_ + "/plain.parquet";
  ASSERT_AND_ASSIGN(auto sink, fs_->OpenOutputStream(parquet_path));
  ASSERT_STATUS_OK(::parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(), sink, 2));
  ASSERT_STATUS_OK(sink->Close());
  std::vector<std::string> paths = {parquet_path};

  ExpectExceptionMessageContainsCode([&]() { PackedRecordBatchReader reader(fs_, paths, schema_, reader_memory_); },
                                     "PackedMetadataCorrupted");
}

}  // namespace milvus_storage
