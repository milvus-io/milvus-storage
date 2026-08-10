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

// Every code lands in one of the eight; the enum is closed.
//
// UNKNOWN is in that set on purpose. It used to be banned from producers, on
// the theory that everything we return must be classified -- but the effect was
// that a failure we could not classify borrowed PERMANENT, which asserts the
// one thing we had not established. "We do not know" is an answer this layer is
// allowed to give.
TEST(ErrorTaxonomyTest, CategoriesAreClosed) {
  for (const auto& row : AllCodes()) {
    switch (row.category) {
      case LOON_ERROR_CATEGORY_UNKNOWN:
      case LOON_ERROR_CATEGORY_USER:
      case LOON_ERROR_CATEGORY_CONFIG:
      case LOON_ERROR_CATEGORY_TRANSIENT:
      case LOON_ERROR_CATEGORY_CONFLICT:
      case LOON_ERROR_CATEGORY_MISSING:
      case LOON_ERROR_CATEGORY_CORRUPTED:
      case LOON_ERROR_CATEGORY_PERMANENT:
        break;
      default:
        FAIL() << row.name << " has category " << row.category << ", which is outside the closed eight";
    }
  }
}

// The same unclassified IO failure must not come back as two different facts
// depending on which layer wrapped it.
//
// WrapExtendError preserves a cause that already carries a classification, and
// stamps the wrapper's own code only when the cause carries none -- so the
// packed codes below appear exclusively for failures nobody classified. If they
// claimed PERMANENT while the plain path answered UNKNOWN for the identical
// condition, a consumer would get opposite facts from two storage paths.
TEST(ErrorTaxonomyTest, UnclassifiedIsUnknownOnEveryPath) {
  const int plain = loon_ffi_error_category(LOON_ARROW_ERROR);
  EXPECT_EQ(plain, LOON_ERROR_CATEGORY_UNKNOWN);

  auto wrapped = WrapExtendError(ExtendStatusCode::PackedStorageIO, "reading packed file",
                                 arrow::Status::IOError("connection reset"));
  auto detail = ExtendStatusDetail::UnwrapStatus(wrapped);
  ASSERT_NE(detail, nullptr) << wrapped.ToString();
  EXPECT_EQ(static_cast<int>(detail->category()), plain)
      << "the packed path reported " << detail->CodeAsString() << " for an unclassified IO failure that the plain "
      << "path calls unknown";

  // ...while a cause that DID arrive classified is never overwritten.
  auto classified = WrapExtendError(ExtendStatusCode::PackedStorageIO, "reading packed file",
                                    MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "denied", "denied"));
  auto kept = ExtendStatusDetail::UnwrapStatus(classified);
  ASSERT_NE(kept, nullptr);
  EXPECT_EQ(kept->code(), ExtendStatusCode::AwsErrorAccessDenied);
}

// PERMANENT is for conditions we WATCHED happen -- our own invariant broken, an
// exception we caught, an allocation that failed. Nothing that merely arrived
// unexplained may claim it, because "permanent" is a statement about the
// future and an unexplained failure supports no statement at all.
TEST(ErrorTaxonomyTest, PermanentIsOnlyForEstablishedFacts) {
  EXPECT_EQ(loon_ffi_error_category(LOON_ARROW_ERROR), LOON_ERROR_CATEGORY_UNKNOWN)
      << "the unclassified fallback claimed a permanent verdict it never established";

  for (int code :
       {LOON_LOGICAL_ERROR, LOON_GOT_EXCEPTION, LOON_UNREACHABLE_ERROR, LOON_MEMORY_ERROR, LOON_INVALID_ARGS}) {
    EXPECT_EQ(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_PERMANENT)
        << loon_ffi_error_name(code) << " is something we saw happen and should say so";
  }
}

// An unrecognized code -- from a producer newer than this consumer -- degrades
// to UNKNOWN, the same answer the producer itself gives for a failure it could
// not explain. Not to PERMANENT: neither situation established anything about
// whether the condition clears.
TEST(ErrorTaxonomyTest, UnrecognizedCodesDegradeToUnknown) {
  for (int code : {-1, 42, 99, 200, 9999}) {
    EXPECT_EQ(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_UNKNOWN) << code;
    EXPECT_NE(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_TRANSIENT) << code;
    EXPECT_EQ(std::string(loon_ffi_error_name(code)), "Unknown error(undefined)") << code;
  }

  EXPECT_EQ(loon_ffi_error_category(LOON_SUCCESS), LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_NE(loon_ffi_error_category(LOON_SUCCESS), LOON_ERROR_CATEGORY_TRANSIENT);
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
    EXPECT_EQ(S3CodeForExtendStatusCode(*code), std::string_view(row.s3_code)) << row.name;
    EXPECT_EQ(ExtendStatusDetail(*code).CodeAsString(), std::string(row.name));
    EXPECT_EQ(static_cast<int>(ExtendStatusDetail(*code).category()), row.category) << row.name;
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
  EXPECT_EQ(loon_ffi_error_category(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)),
            LOON_ERROR_CATEGORY_TRANSIENT);

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

  // The external-table property map is mixed: registered fs.* / writer.*
  // values are deployment configuration, while user extfs.* values are
  // validated later and arrive here as StorageConfigInvalid. A bare
  // InvalidProperties fallback therefore keeps its Config owner.
  EXPECT_EQ(UserSourceErrorCodeFromStatus(arrow::Status::Invalid("x"), LOON_INVALID_PROPERTIES),
            LOON_INVALID_PROPERTIES);
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_PROPERTIES), LOON_ERROR_CATEGORY_CONFIG);

  // Neither re-tag touches retriability: a throttle reached through a
  // user-supplied path is still a throttle.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow", "slow");
  EXPECT_EQ(loon_ffi_error_category(UserSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)),
            LOON_ERROR_CATEGORY_TRANSIENT);
  EXPECT_NE(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_TRANSIENT);
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
  // The fallback that runs when a status carries no classification of ours.
  // It must not reach for a corruption code: of the ~380 unclassified
  // Status::Invalid sites in cpp/src, almost none are corrupt bytes -- they are
  // null-pointer preconditions, missing configuration and caller contract
  // violations. An alert that is mostly false is worse than no alert.
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
    const int code = FFIErrorCodeFromExtendStatus(status, LOON_ARROW_ERROR);
    EXPECT_NE(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_CORRUPTED)
        << status.ToString() << " was called corrupt data without anyone having read a byte";
  }

  // ...while a producer that DID parse the bytes still says so.
  for (auto corrupt : {ExtendStatusCode::PackedMetadataCorrupted, ExtendStatusCode::PackedFileCorrupted,
                       ExtendStatusCode::ManifestCorrupted}) {
    EXPECT_EQ(CategoryForExtendStatusCode(corrupt), ErrorCategory::Corrupted);
  }
}

}  // namespace milvus_storage::test
