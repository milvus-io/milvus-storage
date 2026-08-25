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

TEST_F(ExtendStatusTest, ProducerCanDistinguishPostIoConfigFailure) {
  auto pre_io = MakeExtendError(ExtendStatusCode::StorageConfigInvalid, "invalid endpoint syntax");
  auto post_io = MakeExtendError(ExtendStatusCode::StorageConfigInvalid, arrow::StatusCode::IOError,
                                 "credential endpoint rejected the request");

  EXPECT_TRUE(pre_io.IsInvalid()) << pre_io.ToString();
  EXPECT_TRUE(post_io.IsIOError()) << post_io.ToString();
  auto detail = ExtendStatusDetail::UnwrapStatus(post_io);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid);
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
  // ToString
  {
    ExtendStatusDetail detail(ExtendStatusCode::StorageNoSuchUpload, "my extra");
    auto str = detail.ToString();
    EXPECT_NE(str.find("StorageNoSuchUpload"), std::string::npos);
    EXPECT_NE(str.find("my extra"), std::string::npos);
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
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::PackedUnexpected), milvus::UnexpectedError);
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::StorageAccessDenied), milvus::ConfigInvalid);
}

TEST_F(ExtendStatusTest, PlainArrowStatusFallsBackToCoarseClassification) {
  // No ExtendStatusDetail attached -> coarse arrow status classification.
  //
  // Keep the historical Invalid/Type/Key fallback until the remaining
  // persisted-data producers attach typed details in the final stack layer.
  {
    auto error = ToSegcoreError(arrow::Status::Invalid("some precondition failed"));
    EXPECT_EQ(error.get_error_code(), milvus::DataFormatBroken);
    EXPECT_NE(std::string(error.what()).find("some precondition failed"), std::string::npos);
  }
  // OK remains success.
  {
    auto error = ToSegcoreError(arrow::Status::OK());
    EXPECT_TRUE(error.ok());
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
