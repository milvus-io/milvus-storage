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

// Guards the error taxonomy: every code milvus-storage can return to an upper
// layer is classified exactly once, the two answers a caller needs (whose fault
// / can I retry) never contradict each other, and the FFI view agrees with the
// segcore view.

#include <gtest/gtest.h>

#include <cerrno>
#include <set>
#include <string>
#include <vector>

#include <arrow/status.h>
#include <arrow/util/io_util.h>

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"

namespace milvus_storage::test {
namespace {

struct CodeRow {
  int code;
  const char* name;
  int category;
  const char* s3_code;
  bool is_extend_status;
};

const std::vector<CodeRow>& AllCodes() {
  static const std::vector<CodeRow> rows = {
#define MILVUS_STORAGE_TEST_INTERNAL_ROW(name, code, symbol, category, s3_code) \
  {(code), name, (category), s3_code, false},
    LOON_INTERNAL_ERROR_CODE_LIST(MILVUS_STORAGE_TEST_INTERNAL_ROW)
#undef MILVUS_STORAGE_TEST_INTERNAL_ROW
#define MILVUS_STORAGE_TEST_EXTEND_ROW(name, code, symbol, category, s3_code) \
  {(code), #name, (category), s3_code, true},
        LOON_EXTEND_STATUS_CODE_LIST(MILVUS_STORAGE_TEST_EXTEND_ROW)
#undef MILVUS_STORAGE_TEST_EXTEND_ROW
  };
  return rows;
}

}  // namespace

// Requirement 1: the set of errors returned to upper layers is closed and
// enumerable -- every code has a name, a category and a retry verdict.
TEST(ErrorTaxonomyTest, EveryCodeIsClassified) {
  ASSERT_FALSE(AllCodes().empty());

  for (const auto& row : AllCodes()) {
    EXPECT_NE(row.category, LOON_ERROR_CATEGORY_UNKNOWN) << row.name << " is unclassified";
    EXPECT_EQ(loon_ffi_error_category(row.code), row.category) << row.name;
    EXPECT_EQ(std::string(loon_ffi_error_name(row.code)), std::string(row.name));
    EXPECT_EQ(error_to_string(row.code), std::string(row.name)) << row.name;
  }
}

TEST(ErrorTaxonomyTest, CodeValuesAreUnique) {
  std::set<int> seen;
  for (const auto& row : AllCodes()) {
    EXPECT_TRUE(seen.insert(row.code).second) << "duplicate error code value " << row.code << " (" << row.name << ")";
    EXPECT_NE(row.code, LOON_SUCCESS) << row.name << " collides with LOON_SUCCESS";
  }
}

// Requirement 2 + 3, tied together: retriability is not an independent bit, it
// is the Transient category. This is what keeps "whose fault" and "should I
// retry" from drifting apart.
TEST(ErrorTaxonomyTest, RetryableIsExactlyTransient) {
  for (const auto& row : AllCodes()) {
    const bool retryable = loon_ffi_is_retryable_errcode(row.code) != 0;
    EXPECT_EQ(retryable, row.category == LOON_ERROR_CATEGORY_TRANSIENT) << row.name;

    if (row.category == LOON_ERROR_CATEGORY_USER) {
      EXPECT_FALSE(retryable) << row.name << ": a user error must never be retried";
    }
    if (row.category == LOON_ERROR_CATEGORY_PERMANENT) {
      EXPECT_FALSE(retryable) << row.name << ": a permanent error must never be retried";
    }
  }
}

// An unrecognized code must degrade to "unknown", which consumers treat as
// permanent. Never retry what we cannot classify.
TEST(ErrorTaxonomyTest, UnknownCodesDegradeToPermanentBehaviour) {
  for (int code : {-1, 42, 99, 200, 9999}) {
    EXPECT_EQ(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_UNKNOWN) << code;
    EXPECT_FALSE(loon_ffi_is_retryable_errcode(code)) << code;
    EXPECT_EQ(std::string(loon_ffi_error_name(code)), "Unknown error(undefined)") << code;
  }

  EXPECT_EQ(loon_ffi_error_category(LOON_SUCCESS), LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_SUCCESS));
  EXPECT_EQ(std::string(loon_ffi_error_name(LOON_SUCCESS)), "Success");
}

// The C++ table and the FFI table are generated from the same list; this pins
// that they cannot disagree.
TEST(ErrorTaxonomyTest, ExtendStatusAgreesWithFfiView) {
  for (const auto& row : AllCodes()) {
    auto code = ExtendStatusCodeFromInt(row.code);
    EXPECT_EQ(code.has_value(), row.is_extend_status) << row.name;
    if (!row.is_extend_status) {
      continue;
    }

    EXPECT_EQ(static_cast<int>(CategoryForExtendStatusCode(*code)), row.category) << row.name;
    EXPECT_EQ(DefaultRetryableForExtendStatusCode(*code), loon_ffi_is_retryable_errcode(row.code) != 0) << row.name;
    EXPECT_EQ(S3CodeForExtendStatusCode(*code), std::string_view(row.s3_code)) << row.name;
    EXPECT_EQ(ExtendStatusDetail(*code).CodeAsString(), std::string(row.name));
    EXPECT_EQ(ExtendStatusDetail(*code).retryable(), row.category == LOON_ERROR_CATEGORY_TRANSIENT) << row.name;
    EXPECT_EQ(static_cast<int>(ExtendStatusDetail(*code).category()), row.category) << row.name;
  }
}

// The third table -- the segcore ErrorCode a consumer finally sees -- must
// carry the same verdict. milvus's merr treats 2045 as retriable and 2042 as a
// caller-input error, so a Transient code that mapped to anything but 2045, or
// a User code that mapped to anything but 2042, would silently invert the
// classification at the last hop.
TEST(ErrorTaxonomyTest, SegcoreMappingMatchesCategory) {
  for (const auto& row : AllCodes()) {
    if (!row.is_extend_status) {
      continue;
    }
    auto code = *ExtendStatusCodeFromInt(row.code);
    auto segcore = ToSegcoreErrorCode(code);

    switch (row.category) {
      case LOON_ERROR_CATEGORY_USER:
        EXPECT_EQ(segcore, milvus::InvalidParameter) << row.name << " is a user error but does not map to 2042";
        break;
      case LOON_ERROR_CATEGORY_TRANSIENT:
        EXPECT_EQ(segcore, milvus::StorageTransientError) << row.name << " is transient but does not map to 2045";
        break;
      case LOON_ERROR_CATEGORY_PERMANENT:
        EXPECT_NE(segcore, milvus::StorageTransientError) << row.name << " is permanent but maps to retriable 2045";
        EXPECT_NE(segcore, milvus::InvalidParameter) << row.name << " is permanent but blames the caller";
        break;
      default:
        FAIL() << row.name << " has an unknown category";
    }
  }
}

// The exported constants are the only thing a non-C++ binding can bind to, so
// they must equal the macros the tables were generated from.
TEST(ErrorTaxonomyTest, ExportedConstantsMatchMacros) {
  EXPECT_EQ(loon_errcode_success, LOON_SUCCESS);
  EXPECT_EQ(loon_errcode_invalid_args, LOON_INVALID_ARGS);
  EXPECT_EQ(loon_errcode_memory, LOON_MEMORY_ERROR);
  EXPECT_EQ(loon_errcode_arrow, LOON_ARROW_ERROR);
  EXPECT_EQ(loon_errcode_logical, LOON_LOGICAL_ERROR);
  EXPECT_EQ(loon_errcode_got_exception, LOON_GOT_EXCEPTION);
  EXPECT_EQ(loon_errcode_unreachable, LOON_UNREACHABLE_ERROR);
  EXPECT_EQ(loon_errcode_invalid_properties, LOON_INVALID_PROPERTIES);
  EXPECT_EQ(loon_errcode_fault_inject, LOON_FAULT_INJECT_ERROR);
  EXPECT_EQ(loon_errcode_not_support, LOON_NOT_SUPPORT);
  EXPECT_EQ(loon_errcode_file_not_found, LOON_FILE_NOT_FOUND);
  EXPECT_EQ(loon_errcode_source_not_found, LOON_SOURCE_NOT_FOUND);
  EXPECT_EQ(loon_errcode_source_access_denied, LOON_SOURCE_ACCESS_DENIED);
  EXPECT_EQ(loon_errcode_aws_no_such_upload, LOON_AWS_ERROR_NO_SUCH_UPLOAD);
  EXPECT_EQ(loon_errcode_aws_conflict, LOON_AWS_ERROR_CONFLICT);
  EXPECT_EQ(loon_errcode_aws_precondition_failed, LOON_AWS_ERROR_PRECONDITION_FAILED);
  EXPECT_EQ(loon_errcode_aws_not_found, LOON_AWS_ERROR_NOT_FOUND);
  EXPECT_EQ(loon_errcode_aws_access_denied, LOON_AWS_ERROR_ACCESS_DENIED);
  EXPECT_EQ(loon_errcode_aws_non_retryable, LOON_AWS_ERROR_NON_RETRYABLE);
  EXPECT_EQ(loon_errcode_transient_network, LOON_TRANSIENT_NETWORK);
  EXPECT_EQ(loon_errcode_transient_timeout, LOON_TRANSIENT_TIMEOUT);
  EXPECT_EQ(loon_errcode_transient_throttling, LOON_TRANSIENT_THROTTLING);
  EXPECT_EQ(loon_errcode_transient_service, LOON_TRANSIENT_SERVICE);
  EXPECT_EQ(loon_errcode_txn_exhausted_retry, LOON_TXN_EXHAUSTED_RETRY);
  EXPECT_EQ(loon_errcode_txn_resolution_failed, LOON_TXN_RESOLUTION_FAILED);

  EXPECT_EQ(loon_error_category_unknown, LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_EQ(loon_error_category_user, LOON_ERROR_CATEGORY_USER);
  EXPECT_EQ(loon_error_category_transient, LOON_ERROR_CATEGORY_TRANSIENT);
  EXPECT_EQ(loon_error_category_permanent, LOON_ERROR_CATEGORY_PERMANENT);
}

// The one condition whose owner depends on the call site: a missing object is a
// system failure on an internal path and a user error on a path the user typed.
TEST(ErrorTaxonomyTest, UserSuppliedLocationRetagsNotFoundAndAccessDenied) {
  auto not_found = MakeExtendError(ExtendStatusCode::AwsErrorNotFound, "missing", "missing");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_found, LOON_ARROW_ERROR), LOON_AWS_ERROR_NOT_FOUND);
  EXPECT_EQ(UserSourceErrorCodeFromStatus(not_found, LOON_ARROW_ERROR), LOON_SOURCE_NOT_FOUND);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_NOT_FOUND), LOON_ERROR_CATEGORY_USER);

  auto denied = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "denied", "denied");
  EXPECT_EQ(UserSourceErrorCodeFromStatus(denied, LOON_ARROW_ERROR), LOON_SOURCE_ACCESS_DENIED);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_ACCESS_DENIED), LOON_ERROR_CATEGORY_USER);

  auto enoent = arrow::Status::IOError("missing").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(enoent, LOON_ARROW_ERROR), LOON_FILE_NOT_FOUND);
  EXPECT_EQ(UserSourceErrorCodeFromStatus(enoent, LOON_ARROW_ERROR), LOON_SOURCE_NOT_FOUND);

  // A transient failure stays transient: only ownership is re-tagged, never
  // retriability.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow down", "slow down");
  EXPECT_EQ(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR), LOON_TRANSIENT_THROTTLING);
  EXPECT_TRUE(loon_ffi_is_retryable_errcode(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)));

  // An unclassified status keeps the caller's fallback.
  EXPECT_EQ(UserSourceErrorCodeFromStatus(arrow::Status::Invalid("plain"), LOON_LOGICAL_ERROR), LOON_LOGICAL_ERROR);
}

// Requirement 4: the codes line up with the AWS S3 / Aliyun OSS vocabulary, so
// an operator can map a milvus-storage code to what the object store reported.
// Deliberate divergences are documented in docs/error-codes.md.
TEST(ErrorTaxonomyTest, S3VocabularyIsPinned) {
  auto s3_of = [](ExtendStatusCode code) { return std::string(S3CodeForExtendStatusCode(code)); };

  EXPECT_EQ(s3_of(ExtendStatusCode::AwsErrorNotFound), "NoSuchKey");
  EXPECT_EQ(s3_of(ExtendStatusCode::AwsErrorAccessDenied), "AccessDenied");
  EXPECT_EQ(s3_of(ExtendStatusCode::AwsErrorNoSuchUpload), "NoSuchUpload");
  EXPECT_EQ(s3_of(ExtendStatusCode::AwsErrorConflict), "OperationAborted");
  EXPECT_EQ(s3_of(ExtendStatusCode::AwsErrorPreConditionFailed), "PreconditionFailed");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientThrottling), "SlowDown");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientService), "ServiceUnavailable");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientTimeout), "RequestTimeout");
  EXPECT_EQ(s3_of(ExtendStatusCode::PackedInvalidArgs), "InvalidArgument");

  // Transaction and corruption codes have no object-storage counterpart; an
  // empty string is the documented way to say so.
  EXPECT_EQ(s3_of(ExtendStatusCode::TxnExhaustedRetry), "");
  EXPECT_EQ(s3_of(ExtendStatusCode::TxnResolutionFailed), "");
  EXPECT_EQ(s3_of(ExtendStatusCode::PackedFileCorrupted), "");
}

// Documented divergences from AWS's own client/server split. Pinned so that
// changing one is a deliberate edit to both the table and docs/error-codes.md.
TEST(ErrorTaxonomyTest, DocumentedDivergencesFromAws) {
  // AWS: NoSuchUpload is a 404 client error. Ours: transient -- our retry is at
  // the operation level and starts a fresh multipart upload.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorNoSuchUpload), ErrorCategory::Transient);

  // AWS: NoSuchKey/NoSuchBucket are 4xx client errors. Ours: permanent SYSTEM
  // errors, because on an internal path the caller never chose the key. The
  // user-supplied counterpart is LOON_SOURCE_NOT_FOUND, which IS a user error.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorNotFound), ErrorCategory::Permanent);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_NOT_FOUND), LOON_ERROR_CATEGORY_USER);

  // AWS: AccessDenied is a 403 client error. Ours: permanent system error --
  // the credentials are operator configuration, not part of the request.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorAccessDenied), ErrorCategory::Permanent);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_ACCESS_DENIED), LOON_ERROR_CATEGORY_USER);

  // AWS: InternalError (500) is retriable. Ours: LOON_LOGICAL_ERROR and friends
  // are our own bugs -- retrying reproduces them.
  EXPECT_EQ(loon_ffi_error_category(LOON_LOGICAL_ERROR), LOON_ERROR_CATEGORY_PERMANENT);
  EXPECT_EQ(loon_ffi_error_category(LOON_ARROW_ERROR), LOON_ERROR_CATEGORY_PERMANENT);
}

}  // namespace milvus_storage::test
