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
      // PackedStorageIO: conservatively non-retriable StorageError, but a dormant
      // branch (no live consumer). The retriable 2045 is used by the live
      // no-detail plain-arrow read path, tested separately below.
      {ExtendStatusCode::PackedStorageIO, milvus::StorageError},
      // permanent data corruption
      {ExtendStatusCode::PackedMetadataCorrupted, milvus::DataFormatBroken},
      {ExtendStatusCode::PackedFileCorrupted, milvus::DataFormatBroken},
      // permanent internal storage errors
      {ExtendStatusCode::PackedArrowError, milvus::StorageError},
      {ExtendStatusCode::PackedUnexpected, milvus::StorageError},
      {ExtendStatusCode::AwsErrorNoSuchUpload, milvus::StorageError},
      {ExtendStatusCode::AwsErrorConflict, milvus::StorageError},
      {ExtendStatusCode::AwsErrorPreConditionFailed, milvus::StorageError},
      // permanently-failing S3 errors: must never be transient/2045
      {ExtendStatusCode::AwsErrorNotFound, milvus::StorageError},
      {ExtendStatusCode::AwsErrorAccessDenied, milvus::StorageError},
      {ExtendStatusCode::AwsErrorNonRetryable, milvus::StorageError},
      {ExtendStatusCode::TxnExhaustedRetry, milvus::StorageError},
      {ExtendStatusCode::TxnResolutionFailed, milvus::StorageError},
  };

  for (const auto& test_case : cases) {
    EXPECT_EQ(ToSegcoreErrorCode(test_case.code), test_case.expected);
  }
}

// A Packed* status carries an ExtendStatusDetail, so it is classified by the
// switch. PackedStorageIO is conservatively non-retriable, but this is a DORMANT
// branch (no live consumer -- the packed C-APIs hardcode FileReadFailed/
// FileWriteFailed). Not justified by "v2 retries internally": the S3 SDK retry
// is shared by v2 and v3. (Contrast the live no-detail plain-arrow read path
// below, which stays retriable via 2045 for querynode reroute.)
TEST_F(ExtendStatusTest, PackedStorageIoIsDormantNonRetriable) {
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::PackedStorageIO), milvus::StorageError);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::PackedStorageIO), milvus::StorageTransientError);

  auto status = MakeExtendError(ExtendStatusCode::PackedStorageIO, "object store unavailable", "timeout");
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

// Permanently-failing S3 errors tagged by ErrorToStatus (object/bucket gone,
// bad credentials, SDK-judged non-retryable) must classify permanent, never
// transient/2045 -- otherwise querynode would retry-storm a read that can never
// succeed (retry/reroute hits the same shared object store).
TEST_F(ExtendStatusTest, PermanentS3ErrorsAreNotRetriable) {
  struct Case {
    ExtendStatusCode code;
    const char* name;
  };
  const Case cases[] = {
      {ExtendStatusCode::AwsErrorNotFound, "AwsErrorNotFound"},
      {ExtendStatusCode::AwsErrorAccessDenied, "AwsErrorAccessDenied"},
      {ExtendStatusCode::AwsErrorNonRetryable, "AwsErrorNonRetryable"},
  };
  for (const auto& test_case : cases) {
    auto status = MakeExtendError(test_case.code, "permanent object-store failure", "detail");
    ASSERT_FALSE(status.ok()) << test_case.name;

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << test_case.name;
    EXPECT_EQ(detail->CodeAsString(), test_case.name);

    auto error = ToSegcoreError(status);
    EXPECT_EQ(error.get_error_code(), milvus::StorageError) << test_case.name;
    EXPECT_NE(error.get_error_code(), milvus::StorageTransientError) << test_case.name;
  }
}

TEST_F(ExtendStatusTest, PlainArrowStatusFallsBackToCoarseClassification) {
  // No ExtendStatusDetail attached -> coarse arrow status classification.
  // Plain Invalid means malformed *stored* data here -> permanent corruption.
  {
    auto error = ToSegcoreError(arrow::Status::Invalid("corrupt bytes"));
    EXPECT_EQ(error.get_error_code(), milvus::DataFormatBroken);
    EXPECT_NE(std::string(error.what()).find("corrupt bytes"), std::string::npos);
  }
  // Plain IOError -> retriable StorageTransientError. This is the live read
  // path (FileRowGroupReader / v3 api::Reader / ArrowFileSystem); none retries
  // internally, so a transient IO blip must stay retriable for querynode.
  {
    auto error = ToSegcoreError(arrow::Status::IOError("disk blip"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageTransientError);
    EXPECT_NE(error.get_error_code(), milvus::StorageError);
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
