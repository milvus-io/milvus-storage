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

#include <cerrno>

#include <arrow/status.h>
#include <arrow/util/io_util.h>

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

  auto non_retryable = MakeExtendError(ExtendStatusCode::AwsErrorConflict, "conflict", "detail");
  auto non_retryable_detail = ExtendStatusDetail::UnwrapStatus(non_retryable);
  ASSERT_NE(non_retryable_detail, nullptr);
  EXPECT_FALSE(non_retryable_detail->retryable());

  arrow::Status (*make_extend_error)(ExtendStatusCode, std::string, std::string) = &MakeExtendError;
  auto explicit_three_arg = make_extend_error(ExtendStatusCode::AwsErrorConflict, "conflict", "detail");
  auto explicit_three_arg_detail = ExtendStatusDetail::UnwrapStatus(explicit_three_arg);
  ASSERT_NE(explicit_three_arg_detail, nullptr);
  EXPECT_FALSE(explicit_three_arg_detail->retryable());

  auto retryable = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network failed", "detail");
  auto retryable_detail = ExtendStatusDetail::UnwrapStatus(retryable);
  ASSERT_NE(retryable_detail, nullptr);
  EXPECT_TRUE(retryable_detail->retryable());
  EXPECT_EQ(retryable_detail->code(), ExtendStatusCode::StorageTransientNetwork);
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

TEST_F(ExtendStatusTest, TestExtendStatusCodeRetryability) {
  EXPECT_EQ(ExtendStatusCodeFromInt(50), ExtendStatusCode::PackedInvalidArgs);
  EXPECT_EQ(ExtendStatusCodeFromInt(LOON_AWS_ERROR_NO_SUCH_UPLOAD), ExtendStatusCode::AwsErrorNoSuchUpload);
  EXPECT_EQ(ExtendStatusCodeFromInt(LOON_TRANSIENT_NETWORK), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_FALSE(ExtendStatusCodeFromInt(3).has_value());

  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::PackedInvalidArgs));
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNoSuchUpload));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorConflict));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorPreConditionFailed));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNotFound));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorAccessDenied));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNonRetryable));
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::StorageTransientNetwork));
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::StorageTransientTimeout));
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::StorageTransientThrottling));
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::StorageTransientService));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::TxnExhaustedRetry));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::TxnResolutionFailed));

  auto status = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network", "detail");
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_TRUE(detail->retryable());
}

TEST_F(ExtendStatusTest, TestExtendStatusDetail) {
  // Enum values
  {
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::AwsErrorNoSuchUpload), LOON_AWS_ERROR_NO_SUCH_UPLOAD);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientNetwork), LOON_TRANSIENT_NETWORK);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientTimeout), LOON_TRANSIENT_TIMEOUT);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientThrottling), LOON_TRANSIENT_THROTTLING);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientService), LOON_TRANSIENT_SERVICE);
  }

  // CodeAsString
  {
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorNoSuchUpload).CodeAsString(), "AwsErrorNoSuchUpload");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorConflict).CodeAsString(), "AwsErrorConflict");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::AwsErrorPreConditionFailed).CodeAsString(),
              "AwsErrorPreConditionFailed");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientNetwork).CodeAsString(), "StorageTransientNetwork");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientTimeout).CodeAsString(), "StorageTransientTimeout");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientThrottling).CodeAsString(),
              "StorageTransientThrottling");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientService).CodeAsString(), "StorageTransientService");
  }

  // ToString
  {
    ExtendStatusDetail detail(ExtendStatusCode::AwsErrorNoSuchUpload, "my extra");
    auto str = detail.ToString();
    EXPECT_NE(str.find("AwsErrorNoSuchUpload"), std::string::npos);
    EXPECT_NE(str.find("my extra"), std::string::npos);
  }

  // Retryable
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageTransientNetwork);
    EXPECT_TRUE(detail.retryable());
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
      // branch (no live consumer). The live no-detail plain-arrow read path also
      // maps plain IOError to StorageError, tested separately below.
      {ExtendStatusCode::PackedStorageIO, milvus::StorageError},
      // permanent data corruption
      {ExtendStatusCode::PackedMetadataCorrupted, milvus::DataFormatBroken},
      {ExtendStatusCode::PackedFileCorrupted, milvus::DataFormatBroken},
      // permanent internal storage errors
      {ExtendStatusCode::PackedArrowError, milvus::StorageError},
      {ExtendStatusCode::PackedUnexpected, milvus::StorageError},
      {ExtendStatusCode::AwsErrorNoSuchUpload, milvus::StorageTransientError},
      {ExtendStatusCode::AwsErrorConflict, milvus::StorageError},
      {ExtendStatusCode::AwsErrorPreConditionFailed, milvus::StorageError},
      // permanently-failing S3 errors: must never be transient/2045
      {ExtendStatusCode::AwsErrorNotFound, milvus::ObjectNotExist},
      {ExtendStatusCode::AwsErrorAccessDenied, milvus::StorageError},
      {ExtendStatusCode::AwsErrorNonRetryable, milvus::StorageError},
      {ExtendStatusCode::StorageTransientNetwork, milvus::StorageTransientError},
      {ExtendStatusCode::StorageTransientTimeout, milvus::StorageTransientError},
      {ExtendStatusCode::StorageTransientThrottling, milvus::StorageTransientError},
      {ExtendStatusCode::StorageTransientService, milvus::StorageTransientError},
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
// is shared by v2 and v3. The live no-detail plain-arrow read path below maps
// plain IOError to StorageError/2044 as well.
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
    milvus::ErrorCode expected;
  };
  const Case cases[] = {
      // not-found is fine-grained: ObjectNotExist(2017), still permanent
      {ExtendStatusCode::AwsErrorNotFound, "AwsErrorNotFound", milvus::ObjectNotExist},
      {ExtendStatusCode::AwsErrorAccessDenied, "AwsErrorAccessDenied", milvus::StorageError},
      {ExtendStatusCode::AwsErrorNonRetryable, "AwsErrorNonRetryable", milvus::StorageError},
  };
  for (const auto& test_case : cases) {
    auto status = MakeExtendError(test_case.code, "permanent object-store failure", "detail");
    ASSERT_FALSE(status.ok()) << test_case.name;

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << test_case.name;
    EXPECT_EQ(detail->CodeAsString(), test_case.name);

    auto error = ToSegcoreError(status);
    EXPECT_EQ(error.get_error_code(), test_case.expected) << test_case.name;
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
  // Plain IOError -> non-retriable StorageError. This is the live read path
  // (FileRowGroupReader / v3 api::Reader / ArrowFileSystem); after shared SDK
  // retries, it maps to StorageError/2044.
  {
    auto error = ToSegcoreError(arrow::Status::IOError("disk blip"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_EQ(error.get_error_code(), 2044);
    EXPECT_NE(error.get_error_code(), milvus::StorageTransientError);
  }
  // OOM -> retriable mem-allocate.
  {
    auto error = ToSegcoreError(arrow::Status::OutOfMemory("oom"));
    EXPECT_EQ(error.get_error_code(), milvus::MemAllocateFailed);
  }
  // OK remains success.
  {
    auto error = ToSegcoreError(arrow::Status::OK());
    EXPECT_TRUE(error.ok());
  }
}

TEST_F(ExtendStatusTest, PlainArrowPathNotFoundMapsToObjectNotExist) {
  auto status = arrow::Status::IOError("missing-file").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  ASSERT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);

  auto error = ToSegcoreError(status);

  EXPECT_EQ(error.get_error_code(), milvus::ObjectNotExist);
  EXPECT_NE(error.get_error_code(), milvus::StorageTransientError);
}

TEST_F(ExtendStatusTest, ExtendStatusConvertsToSegcoreError) {
  {
    auto status = MakeExtendError(ExtendStatusCode::PackedFileCorrupted, "bad packed file", "footer mismatch");

    auto error = ToSegcoreError(status);

    EXPECT_EQ(error.get_error_code(), milvus::DataFormatBroken);
    EXPECT_NE(std::string(error.what()).find("bad packed file"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("PackedFileCorrupted"), std::string::npos);
  }
  {
    auto status = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "timeout", "detail");
    auto error = ToSegcoreError(status);

    EXPECT_EQ(error.get_error_code(), milvus::StorageTransientError);
    EXPECT_NE(std::string(error.what()).find("StorageTransientTimeout"), std::string::npos);
  }
  {
    auto status = MakeExtendError(ExtendStatusCode::AwsErrorConflict, "conflict", "detail");
    auto error = ToSegcoreError(status);

    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(std::string(error.what()).find("AwsErrorConflict"), std::string::npos);
  }
}
}  // namespace milvus_storage::test
