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
TEST(ErrorTaxonomyTest, RetryableIsExactlyTheTwoRetriableCategories) {
  for (const auto& row : AllCodes()) {
    const bool retryable = loon_ffi_is_retryable_errcode(row.code) != 0;
    const bool expected = row.category == LOON_ERROR_CATEGORY_TRANSIENT || row.category == LOON_ERROR_CATEGORY_CONFLICT;
    EXPECT_EQ(retryable, expected) << row.name;

    // The two non-retriable categories differ in who fixes it, not in whether
    // to retry -- both must stay out of every retry loop.
    if (row.category == LOON_ERROR_CATEGORY_USER || row.category == LOON_ERROR_CATEGORY_CONFIG) {
      EXPECT_FALSE(retryable) << row.name << ": caller/operator must fix this, retrying cannot";
    }
    if (row.category == LOON_ERROR_CATEGORY_PERMANENT) {
      EXPECT_FALSE(retryable) << row.name << ": a permanent error must never be retried";
    }
  }
}

// Every code must land in one of the seven; the enum is closed. UNKNOWN is a
// consumer-side degradation value and must never be produced.
TEST(ErrorTaxonomyTest, CategoriesAreClosedAndNoProducerEmitsUnknown) {
  for (const auto& row : AllCodes()) {
    switch (row.category) {
      case LOON_ERROR_CATEGORY_USER:
      case LOON_ERROR_CATEGORY_CONFIG:
      case LOON_ERROR_CATEGORY_TRANSIENT:
      case LOON_ERROR_CATEGORY_CONFLICT:
      case LOON_ERROR_CATEGORY_MISSING:
      case LOON_ERROR_CATEGORY_CORRUPTED:
      case LOON_ERROR_CATEGORY_PERMANENT:
        break;
      default:
        FAIL() << row.name << " has category " << row.category << ", which is outside the closed seven";
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
    EXPECT_EQ(RetryableForExtendStatusCode(*code), loon_ffi_is_retryable_errcode(row.code) != 0) << row.name;
    EXPECT_EQ(S3CodeForExtendStatusCode(*code), std::string_view(row.s3_code)) << row.name;
    EXPECT_EQ(ExtendStatusDetail(*code).CodeAsString(), std::string(row.name));
    EXPECT_EQ(ExtendStatusDetail(*code).retryable(),
              row.category == LOON_ERROR_CATEGORY_TRANSIENT || row.category == LOON_ERROR_CATEGORY_CONFLICT)
        << row.name;
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
        // Unreachable by construction, and that IS the invariant. Only an entry
        // point contractually handed a user-supplied location can know the
        // input came from a user, so User is producible exclusively by the
        // re-tagging in exttable_c.cpp -- never by a layer that merely attaches
        // an ExtendStatusDetail, which has no idea who supplied the string.
        // If this fires, someone classified a producer-side code as User and
        // milvus will tell an end user their query is wrong for something they
        // did not do.
        FAIL() << row.name
               << " is an ExtendStatusCode classified User. Producer-side codes cannot know "
                  "whose input failed; classify it Config and let the entry point re-tag.";
        break;
      case LOON_ERROR_CATEGORY_CONFIG:
        // The load-bearing half is the NE: a misconfigured deployment is not
        // the API caller's fault, and reporting it as one sends the user
        // editing their request forever while nobody pages the person who can
        // fix it.
        //
        // The EQ is deliberately a set, not a single value. It used to pin
        // ConfigInvalid alone, which was really "2006 happens to be the only
        // config-shaped code we use". BucketInvalid is equally config-shaped and
        // strictly more precise for a bucket that is not there, so pinning one
        // value would have forced a less accurate mapping to satisfy a test.
        EXPECT_TRUE(segcore == milvus::ConfigInvalid || segcore == milvus::BucketInvalid)
            << row.name << " is a config error but maps to " << segcore << ", which is not config-shaped";
        EXPECT_NE(segcore, milvus::InvalidParameter) << row.name << " blames the caller for an operator problem";
        break;
      case LOON_ERROR_CATEGORY_TRANSIENT:
      case LOON_ERROR_CATEGORY_CONFLICT:
        // Both are retriable, so they share the one retriable segcore code. The
        // strategy difference (plain backoff vs re-read-and-rebase) is carried
        // by the ExtendStatusCode, not by the segcore code.
        EXPECT_EQ(segcore, milvus::StorageTransientError) << row.name << " is retriable but does not map to 2045";
        break;
      case LOON_ERROR_CATEGORY_MISSING:
        // Never 2045. This layer cannot tell a GC race from real data loss, so
        // it refuses to answer the retry question rather than guessing; milvus
        // re-reads the manifest and decides.
        EXPECT_EQ(segcore, milvus::ObjectNotExist) << row.name << " is Missing but does not map to 2017";
        EXPECT_NE(segcore, milvus::StorageTransientError) << row.name << " invents retriability for a missing object";
        EXPECT_FALSE(RetryableForExtendStatusCode(code)) << row.name;
        break;
      case LOON_ERROR_CATEGORY_CORRUPTED:
        EXPECT_EQ(segcore, milvus::DataFormatBroken) << row.name << " is Corrupted but does not map to 2024";
        EXPECT_FALSE(RetryableForExtendStatusCode(code)) << row.name;
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
  EXPECT_EQ(loon_errcode_source_invalid, LOON_SOURCE_INVALID);
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

  EXPECT_EQ(loon_errcode_storage_config_invalid, LOON_STORAGE_CONFIG_INVALID);

  EXPECT_EQ(loon_error_category_unknown, LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_EQ(loon_error_category_user, LOON_ERROR_CATEGORY_USER);
  EXPECT_EQ(loon_error_category_config, LOON_ERROR_CATEGORY_CONFIG);
  EXPECT_EQ(loon_error_category_transient, LOON_ERROR_CATEGORY_TRANSIENT);
  EXPECT_EQ(loon_error_category_conflict, LOON_ERROR_CATEGORY_CONFLICT);
  EXPECT_EQ(loon_error_category_permanent, LOON_ERROR_CATEGORY_PERMANENT);
}

// The one condition whose owner depends on the call site: a missing object is a
// system failure on an internal path and a user error on a path the user typed.
TEST(ErrorTaxonomyTest, UserSuppliedLocationRetagsNotFoundAndAccessDenied) {
  auto not_found = MakeExtendError(ExtendStatusCode::AwsErrorNotFound, "missing", "missing");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_found, LOON_ARROW_ERROR), LOON_AWS_ERROR_NOT_FOUND);
  EXPECT_EQ(UserSourceErrorCodeFromStatus(not_found, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER);

  auto denied = MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "denied", "denied");
  EXPECT_EQ(UserSourceErrorCodeFromStatus(denied, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER);

  auto enoent = arrow::Status::IOError("missing").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(enoent, LOON_ARROW_ERROR), LOON_FILE_NOT_FOUND);
  EXPECT_EQ(UserSourceErrorCodeFromStatus(enoent, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);

  // A transient failure stays transient: only ownership is re-tagged, never
  // retriability.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow down", "slow down");
  EXPECT_EQ(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR), LOON_TRANSIENT_THROTTLING);
  EXPECT_TRUE(loon_ffi_is_retryable_errcode(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)));

  // An unclassified status keeps the caller's fallback.
  EXPECT_EQ(UserSourceErrorCodeFromStatus(arrow::Status::Invalid("plain"), LOON_LOGICAL_ERROR), LOON_LOGICAL_ERROR);
}

// The location spec itself, as opposed to the object it names. These three
// default to Config because a producer cannot tell an operator's milvus.yaml
// from a user's DDL -- but at an entry point that is contractually handed a
// user-supplied location, it can.
//
// This test exists because the mapping was missing and nothing noticed. Merging
// SourceUriInvalid(116) into StorageConfigInvalid(115) silently moved a
// malformed user URI from User/2042 to Config/2006, so a user who typo'd a URI
// in external-table DDL was told to go find an operator. The compensating
// re-tag was promised and not written, and no assertion anywhere would have
// caught it.
TEST(ErrorTaxonomyTest, UserSuppliedLocationRetagsTheLocationSpecItself) {
  struct Case {
    ExtendStatusCode produced;
    int internal_code;
    const char* what;
  };
  const Case cases[] = {
      {ExtendStatusCode::StorageConfigInvalid, LOON_STORAGE_CONFIG_INVALID, "unparseable URI / unusable extfs.*"},
      {ExtendStatusCode::AwsErrorBucketNotFound, LOON_AWS_ERROR_BUCKET_NOT_FOUND, "bucket the user named is gone"},
  };

  for (const auto& c : cases) {
    auto status = MakeExtendError(c.produced, "bad location", "bad location");

    // Off a user-supplied location entry point: User.
    EXPECT_EQ(UserSourceErrorCodeFromStatus(status, LOON_ARROW_ERROR), LOON_SOURCE_INVALID) << c.what;
    EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER) << c.what;

    // Everywhere else the producer's Config verdict stands, because nothing
    // there knows whose string it was.
    EXPECT_EQ(FFIErrorCodeFromExtendStatus(status, LOON_ARROW_ERROR), c.internal_code) << c.what;
    EXPECT_EQ(loon_ffi_error_category(c.internal_code), LOON_ERROR_CATEGORY_CONFIG) << c.what;
  }

  // Properties are part of the same definition -- the credentials and extfs.*
  // keys in an external-source DDL are the user's, and exttable_c.cpp says so
  // in its own comment.
  EXPECT_EQ(UserSourceErrorCodeFromStatus(arrow::Status::Invalid("x"), LOON_INVALID_PROPERTIES), LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_PROPERTIES), LOON_ERROR_CATEGORY_CONFIG);

  // Neither re-tag touches retriability: a throttle reached through a
  // user-supplied path is still a throttle.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow", "slow");
  EXPECT_TRUE(loon_ffi_is_retryable_errcode(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)));
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_SOURCE_INVALID));
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
  // AWS treats 409/412 as terminal client errors. Ours: Conflict, which is
  // retriable -- but only after re-reading. This is the whole reason Conflict
  // is not folded into Transient: a consumer that replays the same conditional
  // write instead of re-reading spins forever.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorConflict), ErrorCategory::Conflict);
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorPreConditionFailed), ErrorCategory::Conflict);
  // A spent retry budget belongs to the loop that spent it and says nothing
  // about an outer attempt made later, in a different contention window.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::TxnExhaustedRetry), ErrorCategory::Conflict);

  // Throttling is retriable like the other transients but must not share their
  // strategy: a plain backoff-and-retry against a throttling store amplifies
  // the overload it is reacting to.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageTransientThrottling), ErrorCategory::Transient);
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageTransientService), ErrorCategory::Transient);

  // AWS: NoSuchKey is a 4xx client error. Ours: Missing -- on an internal path
  // the caller never chose the key, so it is not their fault, and it is not
  // Permanent either because re-reading the manifest may well show the file was
  // legitimately collected. We refuse to answer the retry question rather than
  // guess; milvus re-reads and decides. The user-supplied counterpart is
  // LOON_SOURCE_INVALID, which IS a user error.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorNotFound), ErrorCategory::Missing);
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNotFound));

  // AWS: NoSuchBucket is grouped with NoSuchKey. We split it: nothing was lost,
  // and no re-read produces a bucket. It is a deployment pointing at something
  // that is not there, so Config -- and it lands on BucketInvalid/2016, which
  // milvus already had and we were not using.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorBucketNotFound), ErrorCategory::Config);
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::AwsErrorBucketNotFound), milvus::BucketInvalid);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::AwsErrorBucketNotFound), milvus::ObjectNotExist);

  // AWS: NoSuchUpload is a 404 client error, and we used to call it Conflict and
  // retriable on the theory that our retry starts a fresh upload. That assumed a
  // consumer behaviour this layer cannot guarantee; a resend against the dead
  // upload id fails identically forever.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorNoSuchUpload), ErrorCategory::Missing);
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::AwsErrorNoSuchUpload));
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER);

  // AWS: AccessDenied is a 403 client error. Ours: permanent system error --
  // the credentials are operator configuration, not part of the request.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::AwsErrorAccessDenied), ErrorCategory::Config);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER);

  // AWS: InternalError (500) is retriable. Ours: LOON_LOGICAL_ERROR and friends
  // are our own bugs -- retrying reproduces them.
  EXPECT_EQ(loon_ffi_error_category(LOON_LOGICAL_ERROR), LOON_ERROR_CATEGORY_PERMANENT);
  EXPECT_EQ(loon_ffi_error_category(LOON_ARROW_ERROR), LOON_ERROR_CATEGORY_PERMANENT);

  // Nothing below the FFI entry points may call itself User: only an entry
  // point contractually handed a user-supplied location knows the input came
  // from a user. A property value that will not parse could equally be a
  // milvus.yaml mistake or a segcore hard-coded one, and the producer cannot
  // tell -- so it is Config, and paging an operator for what turns out to be a
  // user typo costs less than telling a user to fix a deployment they cannot
  // touch.
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_PROPERTIES), LOON_ERROR_CATEGORY_CONFIG);
  EXPECT_EQ(loon_ffi_error_category(LOON_STORAGE_CONFIG_INVALID), LOON_ERROR_CATEGORY_CONFIG);
  EXPECT_EQ(loon_ffi_error_category(LOON_NOT_SUPPORT), LOON_ERROR_CATEGORY_CONFIG);
  // The one code that IS User, and the only one: minted exclusively by the
  // re-tagging at loon_exttable_explore / loon_exttable_get_file_info.
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_USER);
  // Caller misuse across the C ABI is a developer's problem, not a user's.
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_ARGS), LOON_ERROR_CATEGORY_PERMANENT);
}

// 2024 DataFormatBroken must have exactly one source: a producer that actually
// parsed the bytes and found them wrong. The coarse arrow-status fallback used
// to guess it from Status::Invalid, which meant that of the ~380 unclassified
// Invalid sites in cpp/src -- null-pointer preconditions, missing config,
// caller contract violations -- every one arrived at segcore claiming the data
// was corrupt. An alert that is mostly false is worse than no alert.
//
// This is the machine-checkable form of the Corrupted discipline: it is easier
// to verify "the fallback never produces 2024" than "only a layer that parsed
// the bytes says Corrupted", and the two are equivalent in effect.
TEST(ErrorTaxonomyTest, CoarseFallbackNeverClaimsCorruption) {
  const arrow::Status unclassified[] = {
      arrow::Status::Invalid("Cannot add null column group"),
      arrow::Status::Invalid("batch schema does not match writer schema"),
      arrow::Status::TypeError("unexpected arrow type"),
      arrow::Status::KeyError("missing key"),
      arrow::Status::IOError("connection reset"),
      arrow::Status::UnknownError("something"),
  };
  for (const auto& status : unclassified) {
    ASSERT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
    auto code = ToSegcoreError(status).get_error_code();
    EXPECT_NE(code, milvus::DataFormatBroken)
        << status.ToString() << " reached segcore claiming corrupt data, without anyone having read a byte";
  }

  // ...while a producer that DID parse the bytes still gets there.
  for (auto corrupt : {ExtendStatusCode::PackedMetadataCorrupted, ExtendStatusCode::PackedFileCorrupted,
                       ExtendStatusCode::ManifestCorrupted}) {
    auto status = MakeExtendError(corrupt, "bad bytes", "bad bytes");
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken);
  }

  // The ENOENT rung is untouched: a filesystem answering "no such file" IS a
  // definitive not-found, which is the Missing discipline, not a guess.
  auto enoent = arrow::Status::IOError("missing").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  EXPECT_EQ(ToSegcoreError(enoent).get_error_code(), milvus::ObjectNotExist);
}

}  // namespace milvus_storage::test
