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
#include <initializer_list>

#include <arrow/status.h>
#include <arrow/util/io_util.h>

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"

namespace milvus_storage::test {

class ExtendStatusTest : public ::testing::Test {};

TEST_F(ExtendStatusTest, TestMakeExtendError) {
  // NoSuchUpload
  {
    auto status = MakeExtendError(ExtendStatusCode::StorageNoSuchUpload, "upload gone", "extra info");
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.IsIOError());

    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageNoSuchUpload);
    EXPECT_EQ(detail->extra_info(), "extra info");
  }

  // Conflict
  {
    auto status = MakeExtendError(ExtendStatusCode::StorageConflict, "conflict occurred", "conflict detail");
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConflict);
  }

  // PreConditionFailed
  {
    auto status = MakeExtendError(ExtendStatusCode::StoragePreConditionFailed, "precondition", "precondition detail");
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StoragePreConditionFailed);
  }

  // Conflict is handled explicitly by business logic, not by generic retry.
  auto conflict = MakeExtendError(ExtendStatusCode::StorageConflict, "conflict", "detail");
  auto conflict_detail = ExtendStatusDetail::UnwrapStatus(conflict);
  ASSERT_NE(conflict_detail, nullptr);
  EXPECT_FALSE(conflict_detail->retryable());
  // PackedMetadataCorrupted stays permanent -- re-reading gives the same bytes.
  auto non_retryable = MakeExtendError(ExtendStatusCode::PackedMetadataCorrupted, "corrupt", "detail");
  auto non_retryable_detail = ExtendStatusDetail::UnwrapStatus(non_retryable);
  ASSERT_NE(non_retryable_detail, nullptr);
  EXPECT_FALSE(non_retryable_detail->retryable());

  arrow::Status (*make_extend_error)(ExtendStatusCode, std::string, std::string) = &MakeExtendError;
  auto explicit_three_arg = make_extend_error(ExtendStatusCode::PackedFileCorrupted, "corrupt", "detail");
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
  EXPECT_EQ(ExtendStatusCodeFromInt(LOON_STORAGE_NO_SUCH_UPLOAD), ExtendStatusCode::StorageNoSuchUpload);
  EXPECT_EQ(ExtendStatusCodeFromInt(LOON_TRANSIENT_NETWORK), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_FALSE(ExtendStatusCodeFromInt(3).has_value());

  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::PackedInvalidArgs));
  // System, not Conflict: resending against a dead upload id fails identically
  // every time. Only a NEW upload helps, and that decision belongs to the layer
  // that owns the write, not to a blind retry here.
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageNoSuchUpload));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageConflict));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StoragePreConditionFailed));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageNotFound));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageAccessDenied));
  EXPECT_TRUE(RetryableForExtendStatusCode(ExtendStatusCode::StorageTransientNetwork));
  EXPECT_TRUE(RetryableForExtendStatusCode(ExtendStatusCode::StorageTransientTimeout));
  EXPECT_TRUE(RetryableForExtendStatusCode(ExtendStatusCode::StorageTransientThrottling));
  EXPECT_TRUE(RetryableForExtendStatusCode(ExtendStatusCode::StorageTransientService));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::TxnExhaustedRetry));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::TxnResolutionFailed));

  auto status = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network", "detail");
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_TRUE(detail->retryable());
}

TEST_F(ExtendStatusTest, TestExtendStatusDetail) {
  // Enum values
  {
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageNoSuchUpload), LOON_STORAGE_NO_SUCH_UPLOAD);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientNetwork), LOON_TRANSIENT_NETWORK);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientTimeout), LOON_TRANSIENT_TIMEOUT);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientThrottling), LOON_TRANSIENT_THROTTLING);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientService), LOON_TRANSIENT_SERVICE);
  }

  // CodeAsString
  {
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageNoSuchUpload).CodeAsString(), "StorageNoSuchUpload");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageConflict).CodeAsString(), "StorageConflict");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StoragePreConditionFailed).CodeAsString(),
              "StoragePreConditionFailed");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientNetwork).CodeAsString(), "StorageTransientNetwork");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientTimeout).CodeAsString(), "StorageTransientTimeout");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientThrottling).CodeAsString(),
              "StorageTransientThrottling");
    EXPECT_EQ(ExtendStatusDetail(ExtendStatusCode::StorageTransientService).CodeAsString(), "StorageTransientService");
  }

  // ToString
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageNoSuchUpload, "my extra");
    auto str = detail.ToString();
    EXPECT_NE(str.find("StorageNoSuchUpload"), std::string::npos);
    EXPECT_NE(str.find("my extra"), std::string::npos);
  }

  // Retryable
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageTransientNetwork);
    EXPECT_TRUE(detail.retryable());
  }

  // SetExtraInfo
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageConflict);
    EXPECT_EQ(detail.extra_info(), "");
    detail.set_extra_info("new info");
    EXPECT_EQ(detail.extra_info(), "new info");
  }

  // TypeId
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageConflict);
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
      {ExtendStatusCode::PackedIO, "PackedIO", false},
      {ExtendStatusCode::PackedMetadataCorrupted, "PackedMetadataCorrupted", false},
      {ExtendStatusCode::PackedFileCorrupted, "PackedFileCorrupted", false},
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
  auto original = MakeExtendError(ExtendStatusCode::PackedIO, "storage failed", "cause");

  auto wrapped = WrapExtendError(ExtendStatusCode::PackedUnexpected, "outer message", original);

  auto detail = ExtendStatusDetail::UnwrapStatus(wrapped);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(wrapped.code(), original.code());
  EXPECT_EQ(wrapped.detail(), original.detail());
  EXPECT_EQ(detail->code(), ExtendStatusCode::PackedIO);
  EXPECT_EQ(detail->extra_info(), "cause");
  EXPECT_NE(wrapped.ToString().find("outer message"), std::string::npos);
  EXPECT_NE(wrapped.ToString().find("storage failed"), std::string::npos);
}

TEST_F(ExtendStatusTest, WrapExtendErrorAddsDetailToPlainStatus) {
  auto wrapped =
      WrapExtendError(ExtendStatusCode::PackedIO, "open packed file", arrow::Status::IOError("disk unavailable"));

  auto detail = ExtendStatusDetail::UnwrapStatus(wrapped);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::PackedIO);
  EXPECT_NE(wrapped.ToString().find("open packed file"), std::string::npos);
  EXPECT_NE(detail->extra_info().find("disk unavailable"), std::string::npos);
}

TEST_F(ExtendStatusTest, WrapExtendErrorPreservesErrnoDetail) {
  auto cause = arrow::Status::IOError("missing-file").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));

  auto wrapped = WrapExtendError(ExtendStatusCode::PackedIO, "open packed file", cause);

  EXPECT_EQ(wrapped.code(), cause.code());
  EXPECT_EQ(wrapped.detail(), cause.detail());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(wrapped), ENOENT);
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(wrapped), nullptr);
  EXPECT_NE(wrapped.ToString().find("open packed file"), std::string::npos);
  EXPECT_NE(wrapped.ToString().find("missing-file"), std::string::npos);
  EXPECT_EQ(ToSegcoreError(wrapped).get_error_code(), milvus::ObjectNotExist);
}

TEST_F(ExtendStatusTest, WrapExtendErrorPreservesOutOfMemory) {
  auto wrapped = WrapExtendError(ExtendStatusCode::PackedMetadataCorrupted, "parse packed metadata",
                                 arrow::Status::OutOfMemory("allocation failed"));

  EXPECT_TRUE(wrapped.IsOutOfMemory()) << wrapped.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(wrapped), nullptr);
  EXPECT_NE(wrapped.message().find("parse packed metadata"), std::string::npos);
  EXPECT_NE(wrapped.message().find("allocation failed"), std::string::npos);
  EXPECT_EQ(ToSegcoreError(wrapped).get_error_code(), milvus::MemAllocateFailed);
}

TEST_F(ExtendStatusTest, ExtendCodesMapToSegcoreErrorCode) {
  struct Case {
    ExtendStatusCode code;
    milvus::ErrorCode expected;
  };

  const Case cases[] = {
      // input (non-retriable)
      // Internal API misuse, not an end user's parameter: 2044, not 2042.
      // Internal API misuse is a defect report, not a storage verdict.
      {ExtendStatusCode::PackedInvalidArgs, milvus::UnexpectedError},
      // Direct C++/segcore consumers receive StorageError. FFI consumers retain
      // PackedIO and classify it from the exported error table.
      {ExtendStatusCode::PackedIO, milvus::StorageError},
      // data-format failures
      {ExtendStatusCode::PackedMetadataCorrupted, milvus::DataFormatBroken},
      {ExtendStatusCode::PackedFileCorrupted, milvus::DataFormatBroken},
      // System/internal storage errors
      {ExtendStatusCode::PackedUnexpected, milvus::UnexpectedError},
      {ExtendStatusCode::InternalInvariantViolated, milvus::UnexpectedError},
      {ExtendStatusCode::StorageNoSuchUpload, milvus::StorageError},
      {ExtendStatusCode::StorageConflict, milvus::StorageError},
      {ExtendStatusCode::StoragePreConditionFailed, milvus::StorageError},
      // non-retryable S3 errors: must never land on 2045
      {ExtendStatusCode::StorageNotFound, milvus::ObjectNotExist},
      // System: the credentials are the operator's, so this has to reach whoever
      // owns the deployment rather than be filed as a generic storage failure.
      {ExtendStatusCode::StorageAccessDenied, milvus::ConfigInvalid},
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

// A Packed* status carries an ExtendStatusDetail. Direct C++ consumers map it
// through ToSegcoreErrorCode; FFI consumers retain the exact Packed* code and
// derive the same category from the exported table.
TEST_F(ExtendStatusTest, PackedIoParentsHaveDistinctRetryBehaviour) {
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::PackedIO), milvus::StorageError);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::PackedIO), milvus::StorageTransientError);

  auto status = MakeExtendError(ExtendStatusCode::PackedIO, "object store unavailable", "timeout");
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

// Permanently-failing S3 errors tagged by ErrorToStatus (object/bucket gone or
// bad credentials) must classify permanent, never transient/2045 -- otherwise
// querynode would retry-storm a read that can never succeed (retry/reroute hits
// the same shared object store).
TEST_F(ExtendStatusTest, PermanentS3ErrorsAreNotRetriable) {
  struct Case {
    ExtendStatusCode code;
    const char* name;
    milvus::ErrorCode expected;
  };
  const Case cases[] = {
      // not-found is fine-grained: ObjectNotExist(2017), still permanent
      {ExtendStatusCode::StorageNotFound, "StorageNotFound", milvus::ObjectNotExist},
      {ExtendStatusCode::StorageAccessDenied, "StorageAccessDenied", milvus::ConfigInvalid},
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
  //
  // A plain Invalid does NOT mean malformed stored data, which is what this
  // test used to assert. Of the ~380 unclassified Status::Invalid sites in
  // cpp/src, almost none are corrupt bytes -- they are null-pointer
  // preconditions, missing configuration and caller contract violations.
  // Claiming corruption for all of them made 2024 an alert nobody could trust.
  {
    auto error = ToSegcoreError(arrow::Status::Invalid("some precondition failed"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(error.get_error_code(), milvus::DataFormatBroken);
    EXPECT_NE(std::string(error.what()).find("some precondition failed"), std::string::npos);
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
  // OOM -> the node could not allocate, which segcore has a code for. Filed as
  // a storage failure it pointed at an object store that never saw the request.
  // Still never a retry promise: this layer cannot promise a replay finds more
  // memory.
  {
    auto error = ToSegcoreError(arrow::Status::OutOfMemory("oom"));
    EXPECT_EQ(error.get_error_code(), milvus::MemAllocateFailed);
    EXPECT_NE(error.get_error_code(), milvus::StorageTransientError);
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
    auto status = MakeExtendError(ExtendStatusCode::StorageConflict, "conflict", "detail");
    auto error = ToSegcoreError(status);

    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(std::string(error.what()).find("StorageConflict"), std::string::npos);
  }
}
}  // namespace milvus_storage::test
