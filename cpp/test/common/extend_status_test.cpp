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

#include <gtest/gtest.h>

#include <arrow/status.h>

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"

namespace milvus_storage::test {

class ExtendStatusTest : public ::testing::Test {};

TEST_F(ExtendStatusTest, TestMakeExtendError) {
  // NoSuchUpload
  {
    auto status = MakeExtendError(ExtendStatusCode::AwsErrorNoSuchUpload, "upload gone", "extra info");
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.IsIOError());

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNoSuchUpload);
    EXPECT_EQ(detail->extra_info(), "extra info");
  }

  // Conflict
  {
    auto status = MakeExtendError(ExtendStatusCode::AwsErrorConflict, "conflict occurred", "conflict detail");
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorConflict);
  }

  // PreConditionFailed
  {
    auto status = MakeExtendError(ExtendStatusCode::AwsErrorPreConditionFailed, "precondition", "precondition detail");
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorPreConditionFailed);
  }
}

TEST_F(ExtendStatusTest, TestUnwrapStatus) {
  // Plain IOError → nullptr
  {
    auto detail = ExtendStatusDetail::UnwrapStatus(arrow::Status::IOError("plain error"));
    EXPECT_EQ(detail, nullptr);
  }

  // OK status → nullptr
  {
    auto detail = ExtendStatusDetail::UnwrapStatus(arrow::Status::OK());
    EXPECT_EQ(detail, nullptr);
  }
}

TEST_F(ExtendStatusTest, TestExtendStatusDetail) {
  // CodeAsString
  {
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorNoSuchUpload).CodeAsString(), "AwsErrorNoSuchUpload");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorConflict).CodeAsString(), "AwsErrorConflict");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorPreConditionFailed).CodeAsString(),
              "AwsErrorPreConditionFailed");
  }

  // ToString
  {
    ExtendStatusDetail detail(ExtendStatusCode::AwsErrorNoSuchUpload, "my extra");
    auto str = detail.ToString();
    EXPECT_NE(str.find("AwsErrorNoSuchUpload"), std::string::npos);
    EXPECT_NE(str.find("my extra"), std::string::npos);
  }

  // SetExtraInfo
  {
    ExtendStatusDetail detail(ExtendStatusCode::AwsErrorConflict);
    EXPECT_EQ(detail.extra_info(), "");
    detail.set_extra_info("new info");
    EXPECT_EQ(detail.extra_info(), "new info");
  }

  // TypeId
  {
    ExtendStatusDetail detail(ExtendStatusCode::AwsErrorConflict);
    EXPECT_NE(detail.type_id(), nullptr);
    EXPECT_EQ(std::string(detail.type_id()), "milvus_storage::ExtendStatusDetail");
  }
}

TEST_F(ExtendStatusTest, PackedCodesUseExpectedArrowStatusCodeAndDetail) {
  struct Case {
    ExtendStatusCode code;
    const char* name;
    bool is_invalid;
  };

  const Case cases[] = {
      {ExtendStatusCode::PackedInvalidArgs, "PackedInvalidArgs", true},
      {ExtendStatusCode::PackedStorageIO, "PackedStorageIO", false},
      {ExtendStatusCode::PackedMetadataCorrupted, "PackedMetadataCorrupted", false},
      {ExtendStatusCode::PackedFileCorrupted, "PackedFileCorrupted", false},
      {ExtendStatusCode::PackedArrowError, "PackedArrowError", false},
      {ExtendStatusCode::PackedUnexpected, "PackedUnexpected", false},
  };

  for (const auto& test_case : cases) {
    auto status = MakeExtendError(test_case.code, "message", "extra");
    ASSERT_FALSE(status.ok()) << test_case.name;
    EXPECT_EQ(status.IsInvalid(), test_case.is_invalid) << test_case.name;
    EXPECT_EQ(status.IsIOError(), !test_case.is_invalid) << test_case.name;

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << test_case.name << ": " << status.ToString();
    EXPECT_EQ(detail->code(), test_case.code);
    EXPECT_EQ(detail->extra_info(), "extra");
    EXPECT_EQ(detail->CodeAsString(), test_case.name);
    EXPECT_NE(detail->ToString().find(test_case.name), std::string::npos);
    EXPECT_NE(detail->ToString().find("extra"), std::string::npos);
  }
}

TEST_F(ExtendStatusTest, WrapExtendErrorPreservesExistingDetail) {
  auto original = MakeExtendError(ExtendStatusCode::PackedStorageIO, "storage failed", "cause");

  auto wrapped = WrapExtendError(ExtendStatusCode::PackedUnexpected, "outer message", original);

  auto detail = ExtendStatusDetail::UnwrapStatus(wrapped);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::PackedStorageIO);
  EXPECT_NE(wrapped.ToString().find("outer message"), std::string::npos);
  EXPECT_NE(wrapped.ToString().find("storage failed"), std::string::npos);
  EXPECT_NE(detail->extra_info().find("storage failed"), std::string::npos);
}

TEST_F(ExtendStatusTest, WrapExtendErrorAddsDetailToPlainStatus) {
  auto wrapped = WrapExtendError(ExtendStatusCode::PackedStorageIO, "open packed file",
                                 arrow::Status::IOError("disk unavailable"));

  auto detail = ExtendStatusDetail::UnwrapStatus(wrapped);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::PackedStorageIO);
  EXPECT_NE(wrapped.ToString().find("open packed file"), std::string::npos);
  EXPECT_NE(detail->extra_info().find("disk unavailable"), std::string::npos);
}

TEST_F(ExtendStatusTest, ExtendCodesMapToSegcoreErrorCode) {
  struct Case {
    ExtendStatusCode code;
    milvus::ErrorCode expected;
  };

  const Case cases[] = {
      // input (non-retriable)
      {ExtendStatusCode::PackedInvalidArgs, milvus::InvalidParameter},
      // retriable storage IO -- must NOT collapse into StorageError
      {ExtendStatusCode::PackedStorageIO, milvus::StorageTransientError},
      // permanent data corruption
      {ExtendStatusCode::PackedMetadataCorrupted, milvus::DataFormatBroken},
      {ExtendStatusCode::PackedFileCorrupted, milvus::DataFormatBroken},
      // permanent internal storage errors
      {ExtendStatusCode::PackedArrowError, milvus::StorageError},
      {ExtendStatusCode::PackedUnexpected, milvus::StorageError},
      {ExtendStatusCode::AwsErrorNoSuchUpload, milvus::StorageError},
      {ExtendStatusCode::AwsErrorConflict, milvus::StorageError},
      {ExtendStatusCode::AwsErrorPreConditionFailed, milvus::StorageError},
      {ExtendStatusCode::TxnExhaustedRetry, milvus::StorageError},
      {ExtendStatusCode::TxnResolutionFailed, milvus::StorageError},
  };

  for (const auto& test_case : cases) {
    EXPECT_EQ(ToSegcoreErrorCode(test_case.code), test_case.expected);
  }
}

// The retriable verdict is the load-bearing property: a transient storage IO
// failure must stay retriable and never collapse into the non-retriable
// StorageError fallback.
TEST_F(ExtendStatusTest, StorageIoMapsToRetriableTransientCode) {
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::PackedStorageIO), milvus::StorageTransientError);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::PackedStorageIO), milvus::StorageError);

  auto status = MakeExtendError(ExtendStatusCode::PackedStorageIO, "object store unavailable", "timeout");
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageTransientError);
}

TEST_F(ExtendStatusTest, PlainArrowStatusFallsBackToCoarseClassification) {
  // No ExtendStatusDetail attached -> coarse arrow status classification.
  // Plain Invalid means malformed *stored* data here -> permanent corruption.
  {
    auto error = ToSegcoreError(arrow::Status::Invalid("corrupt bytes"));
    EXPECT_EQ(error.get_error_code(), milvus::DataFormatBroken);
    EXPECT_NE(std::string(error.what()).find("corrupt bytes"), std::string::npos);
  }
  // Plain IOError -> retriable transient storage error.
  {
    auto error = ToSegcoreError(arrow::Status::IOError("disk blip"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageTransientError);
  }
  // OOM -> retriable mem-allocate.
  {
    auto error = ToSegcoreError(arrow::Status::OutOfMemory("oom"));
    EXPECT_EQ(error.get_error_code(), milvus::MemAllocateFailed);
  }
}

TEST_F(ExtendStatusTest, ExtendStatusConvertsToSegcoreError) {
  auto status = MakeExtendError(ExtendStatusCode::PackedFileCorrupted, "bad packed file", "footer mismatch");

  auto error = ToSegcoreError(status);

  EXPECT_EQ(error.get_error_code(), milvus::DataFormatBroken);
  EXPECT_NE(std::string(error.what()).find("bad packed file"), std::string::npos);
  EXPECT_NE(std::string(error.what()).find("PackedFileCorrupted"), std::string::npos);
}

}  // namespace milvus_storage::test
