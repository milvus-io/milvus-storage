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

  // AwsErrorConflict is Conflict class: recoverable, but only by a re-read and
  // rebase, which is not what `retryable` promises. The category is what a
  // consumer capable of rebasing looks at.
  auto conflict = MakeExtendError(ExtendStatusCode::AwsErrorConflict, "conflict", "detail");
  auto conflict_detail = ExtendStatusDetail::UnwrapStatus(conflict);
  ASSERT_NE(conflict_detail, nullptr);
  EXPECT_EQ(conflict_detail->category(), ErrorCategory::Conflict);
  EXPECT_NE(conflict_detail->category(), ErrorCategory::Transient);
  // PackedMetadataCorrupted stays permanent -- re-reading gives the same bytes.
  auto non_retryable = MakeExtendError(ExtendStatusCode::PackedMetadataCorrupted, "corrupt", "detail");
  auto non_retryable_detail = ExtendStatusDetail::UnwrapStatus(non_retryable);
  ASSERT_NE(non_retryable_detail, nullptr);
  EXPECT_NE(non_retryable_detail->category(), ErrorCategory::Transient);

  arrow::Status (*make_extend_error)(ExtendStatusCode, std::string, std::string) = &MakeExtendError;
  auto explicit_three_arg = make_extend_error(ExtendStatusCode::PackedFileCorrupted, "corrupt", "detail");
  auto explicit_three_arg_detail = ExtendStatusDetail::UnwrapStatus(explicit_three_arg);
  ASSERT_NE(explicit_three_arg_detail, nullptr);
  EXPECT_NE(explicit_three_arg_detail->category(), ErrorCategory::Transient);

  auto retryable = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network failed", "detail");
  auto retryable_detail = ExtendStatusDetail::UnwrapStatus(retryable);
  ASSERT_NE(retryable_detail, nullptr);
  EXPECT_EQ(retryable_detail->category(), ErrorCategory::Transient);
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
    EXPECT_EQ(detail.category(), ErrorCategory::Transient);
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
  EXPECT_EQ(wrapped.code(), original.code());
  EXPECT_EQ(wrapped.detail(), original.detail());
  EXPECT_EQ(detail->code(), ExtendStatusCode::PackedStorageIO);
  EXPECT_EQ(detail->extra_info(), "cause");
  EXPECT_NE(wrapped.ToString().find("outer message"), std::string::npos);
  EXPECT_NE(wrapped.ToString().find("storage failed"), std::string::npos);
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

TEST_F(ExtendStatusTest, WrapExtendErrorPreservesErrnoDetail) {
  auto cause = arrow::Status::IOError("missing-file").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));

  auto wrapped = WrapExtendError(ExtendStatusCode::PackedStorageIO, "open packed file", cause);

  EXPECT_EQ(wrapped.code(), cause.code());
  EXPECT_EQ(wrapped.detail(), cause.detail());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(wrapped), ENOENT);
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(wrapped), nullptr);
  EXPECT_NE(wrapped.ToString().find("open packed file"), std::string::npos);
  EXPECT_NE(wrapped.ToString().find("missing-file"), std::string::npos);
}

// S3 failures that cannot clear on their own must not be classified Transient:
// a retry or a reroute hits the same shared object store, so a wrong Transient
// here is a retry storm against a read that can never succeed.
TEST_F(ExtendStatusTest, PermanentS3ErrorsNeverLookTransient) {
  for (auto code : {ExtendStatusCode::AwsErrorNotFound, ExtendStatusCode::AwsErrorAccessDenied,
                    ExtendStatusCode::PackedStorageIO}) {
    auto status = MakeExtendError(code, "permanent object-store failure", "detail");
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), code);
    EXPECT_NE(detail->category(), ErrorCategory::Transient) << detail->CodeAsString();
  }
}

}  // namespace milvus_storage::test
