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
  EXPECT_EQ(ExtendStatusCodeFromInt(50), ExtendStatusCode::AwsErrorNoSuchUpload);
  EXPECT_EQ(ExtendStatusCodeFromInt(60), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_FALSE(ExtendStatusCodeFromInt(3).has_value());

  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNoSuchUpload));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorConflict));
  EXPECT_FALSE(DefaultRetryableForExtendStatusCode(ExtendStatusCode::AwsErrorPreConditionFailed));
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
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientNetwork), 60);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientTimeout), 61);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientThrottling), 62);
    EXPECT_EQ(static_cast<int>(ExtendStatusCode::StorageTransientService), 63);
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

TEST_F(ExtendStatusTest, ExtendCodesMapToSegcoreErrorCode) {
  struct Case {
    ExtendStatusCode code;
    int expected;
  };

  const Case cases[] = {
      {ExtendStatusCode::AwsErrorNoSuchUpload, static_cast<int>(milvus::StorageTransientError)},
      {ExtendStatusCode::AwsErrorConflict, static_cast<int>(milvus::StorageError)},
      {ExtendStatusCode::AwsErrorPreConditionFailed, static_cast<int>(milvus::StorageError)},
      {ExtendStatusCode::StorageTransientNetwork, static_cast<int>(milvus::StorageTransientError)},
      {ExtendStatusCode::StorageTransientTimeout, static_cast<int>(milvus::StorageTransientError)},
      {ExtendStatusCode::StorageTransientThrottling, static_cast<int>(milvus::StorageTransientError)},
      {ExtendStatusCode::StorageTransientService, static_cast<int>(milvus::StorageTransientError)},
      {ExtendStatusCode::TxnExhaustedRetry, static_cast<int>(milvus::StorageError)},
      {ExtendStatusCode::TxnResolutionFailed, static_cast<int>(milvus::StorageError)},
  };

  for (const auto& test_case : cases) {
    EXPECT_EQ(ToSegcoreErrorCode(test_case.code), test_case.expected);
  }
}

TEST_F(ExtendStatusTest, ConvertsExtendStatusToSegcoreError) {
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

TEST_F(ExtendStatusTest, PlainArrowStatusFallsBackToStorageError) {
  {
    auto error = ToSegcoreError(arrow::Status::OK());
    EXPECT_TRUE(error.ok());
  }
  {
    auto error = ToSegcoreError(arrow::Status::OutOfMemory("oom"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(std::string(error.what()).find("oom"), std::string::npos);
  }
  {
    auto error = ToSegcoreError(arrow::Status::IOError("disk blip"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(std::string(error.what()).find("disk blip"), std::string::npos);
  }
  {
    auto error = ToSegcoreError(arrow::Status::Invalid("bad bytes"));
    EXPECT_EQ(error.get_error_code(), milvus::StorageError);
    EXPECT_NE(std::string(error.what()).find("bad bytes"), std::string::npos);
  }
}

}  // namespace milvus_storage::test
