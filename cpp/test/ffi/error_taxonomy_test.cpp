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
// layer is classified exactly once, generic handling and retry answers never
// contradict each other, and the FFI view agrees with the segcore view.

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
// enumerable -- every code has a name and one handling category; the transient
// retry hint is derived from that category rather than stored independently.
TEST(ErrorTaxonomyTest, EveryCodeIsClassified) {
  ASSERT_FALSE(AllCodes().empty());

  for (const auto& row : AllCodes()) {
    EXPECT_NE(row.category, LOON_ERROR_CATEGORY_UNKNOWN) << row.name << " is unclassified";
    EXPECT_EQ(loon_ffi_error_category(row.code), row.category) << row.name;
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

TEST(ErrorTaxonomyTest, RetryableCategoryControlsOnlyTheTransientHint) {
  for (const auto& row : AllCodes()) {
    const bool retryable = loon_ffi_is_retryable_errcode(row.code) != 0;
    const bool expected = row.category == LOON_ERROR_CATEGORY_RETRYABLE;
    EXPECT_EQ(retryable, expected) << row.name;
  }
}

// Every code must land in one of the five; UNKNOWN is a
// consumer-side degradation value and must never be produced.
TEST(ErrorTaxonomyTest, CategoriesAreClosedAndNoProducerEmitsUnknown) {
  for (const auto& row : AllCodes()) {
    switch (row.category) {
      case LOON_ERROR_CATEGORY_USER:
      case LOON_ERROR_CATEGORY_RETRYABLE:
      case LOON_ERROR_CATEGORY_CONFLICT:
      case LOON_ERROR_CATEGORY_DATA_FORMAT:
      case LOON_ERROR_CATEGORY_SYSTEM:
        break;
      default:
        FAIL() << row.name << " has category " << row.category << ", which is outside the closed five";
    }
  }
}

// An unrecognized code must degrade to "unknown", which carries no generic
// retry hint. An operation owner can still apply its own idempotency policy.
TEST(ErrorTaxonomyTest, UnknownCodesDegradeWithoutTransientHint) {
  for (int code : {-1, 42, 99, 200, 9999}) {
    EXPECT_EQ(loon_ffi_error_category(code), LOON_ERROR_CATEGORY_UNKNOWN) << code;
    EXPECT_FALSE(loon_ffi_is_retryable_errcode(code)) << code;
  }

  EXPECT_EQ(loon_ffi_error_category(LOON_SUCCESS), LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_SUCCESS));
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
    EXPECT_EQ(ExtendStatusDetail(*code).retryable(), row.category == LOON_ERROR_CATEGORY_RETRYABLE) << row.name;
  }
}

// Segcore has only generic transient/non-transient storage codes. Conflict must
// not use the transient code; only a conflict-aware consumer can decide how to
// re-read/rebase and submit new work.
TEST(ErrorTaxonomyTest, SegcoreMappingMatchesCategory) {
  for (const auto& row : AllCodes()) {
    if (!row.is_extend_status) {
      continue;
    }
    auto code = *ExtendStatusCodeFromInt(row.code);
    auto segcore = ToSegcoreErrorCode(code);

    switch (row.category) {
      case LOON_ERROR_CATEGORY_USER:
        FAIL() << row.name
               << " is an ExtendStatusCode classified User. Producer-side codes cannot know "
                  "whose input failed; classify it System and let the entry point re-tag.";
        break;
      case LOON_ERROR_CATEGORY_RETRYABLE:
        EXPECT_EQ(segcore, milvus::StorageTransientError) << row.name << " is retriable but does not map to 2045";
        break;
      case LOON_ERROR_CATEGORY_CONFLICT:
        EXPECT_EQ(segcore, milvus::StorageError) << row.name << " exposes conflict as a generic retry";
        EXPECT_FALSE(RetryableForExtendStatusCode(code)) << row.name;
        break;
      case LOON_ERROR_CATEGORY_DATA_FORMAT:
        EXPECT_EQ(segcore, milvus::DataFormatBroken) << row.name << " is DataFormat but does not map to 2024";
        EXPECT_FALSE(RetryableForExtendStatusCode(code)) << row.name;
        break;
      case LOON_ERROR_CATEGORY_SYSTEM:
        EXPECT_NE(segcore, milvus::StorageTransientError) << row.name << " is System but maps to retriable 2045";
        EXPECT_NE(segcore, milvus::InvalidParameter) << row.name << " is System but blames the caller";
        break;
      default:
        FAIL() << row.name << " has an unknown category";
    }
  }
}

// Error-code symbols are exported; category values are compile-time C enum
// constants so consumers can use them in switch statements.
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
  EXPECT_EQ(loon_errcode_user_invalid_argument, LOON_USER_INVALID_ARGUMENT);
  EXPECT_EQ(loon_errcode_file_not_found, LOON_FILE_NOT_FOUND);
  EXPECT_EQ(loon_errcode_source_invalid, LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_errcode_packed_invalid_args, static_cast<int>(ExtendStatusCode::PackedInvalidArgs));
  EXPECT_EQ(loon_errcode_packed_io, static_cast<int>(ExtendStatusCode::PackedIO));
  EXPECT_EQ(loon_errcode_packed_metadata_corrupted, static_cast<int>(ExtendStatusCode::PackedMetadataCorrupted));
  EXPECT_EQ(loon_errcode_packed_file_corrupted, static_cast<int>(ExtendStatusCode::PackedFileCorrupted));
  EXPECT_EQ(loon_errcode_packed_unexpected, static_cast<int>(ExtendStatusCode::PackedUnexpected));
  EXPECT_EQ(loon_errcode_storage_no_such_upload, LOON_STORAGE_NO_SUCH_UPLOAD);
  EXPECT_EQ(loon_errcode_storage_conflict, LOON_STORAGE_CONFLICT);
  EXPECT_EQ(loon_errcode_storage_precondition_failed, LOON_STORAGE_PRECONDITION_FAILED);
  EXPECT_EQ(loon_errcode_storage_not_found, LOON_STORAGE_NOT_FOUND);
  EXPECT_EQ(loon_errcode_storage_access_denied, LOON_STORAGE_ACCESS_DENIED);
  EXPECT_EQ(loon_errcode_transient_network, LOON_TRANSIENT_NETWORK);
  EXPECT_EQ(loon_errcode_transient_timeout, LOON_TRANSIENT_TIMEOUT);
  EXPECT_EQ(loon_errcode_transient_throttling, LOON_TRANSIENT_THROTTLING);
  EXPECT_EQ(loon_errcode_transient_service, LOON_TRANSIENT_SERVICE);
  EXPECT_EQ(loon_errcode_txn_exhausted_retry, LOON_TXN_EXHAUSTED_RETRY);
  EXPECT_EQ(loon_errcode_txn_resolution_failed, LOON_TXN_RESOLUTION_FAILED);

  EXPECT_EQ(loon_errcode_storage_config_invalid, LOON_STORAGE_CONFIG_INVALID);

  EXPECT_EQ(loon_error_category_unknown, LOON_ERROR_CATEGORY_UNKNOWN);
  EXPECT_EQ(loon_error_category_user, LOON_ERROR_CATEGORY_USER);
  EXPECT_EQ(loon_error_category_retryable, LOON_ERROR_CATEGORY_RETRYABLE);
  EXPECT_EQ(loon_error_category_conflict, LOON_ERROR_CATEGORY_CONFLICT);
  EXPECT_EQ(loon_error_category_data_format, LOON_ERROR_CATEGORY_DATA_FORMAT);
  EXPECT_EQ(loon_error_category_system, LOON_ERROR_CATEGORY_SYSTEM);
}

TEST(ErrorTaxonomyTest, ExternalSourceMapsTerminalAccessFailures) {
  auto not_found = MakeExtendError(ExtendStatusCode::StorageNotFound, "missing", "missing");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_found, LOON_ARROW_ERROR), LOON_STORAGE_NOT_FOUND);
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(not_found, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM);

  auto denied = MakeExtendError(ExtendStatusCode::StorageAccessDenied, "denied", "denied");
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(denied, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM);

  auto enoent = arrow::Status::IOError("missing").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(enoent, LOON_ARROW_ERROR), LOON_FILE_NOT_FOUND);
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(enoent, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);

  // A transient failure stays transient.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow down", "slow down");
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR), LOON_TRANSIENT_THROTTLING);
  EXPECT_TRUE(loon_ffi_is_retryable_errcode(ExternalSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)));

  // An unclassified status keeps the caller's fallback.
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(arrow::Status::Invalid("plain"), LOON_LOGICAL_ERROR), LOON_LOGICAL_ERROR);
}

TEST(ErrorTaxonomyTest, ExternalSourceMapsLocationResolutionFailures) {
  struct Case {
    ExtendStatusCode produced;
    int internal_code;
    const char* what;
  };
  const Case cases[] = {
      {ExtendStatusCode::StorageConfigInvalid, LOON_STORAGE_CONFIG_INVALID, "unparseable URI / unusable extfs.*"},
      {ExtendStatusCode::StorageBucketNotFound, LOON_STORAGE_BUCKET_NOT_FOUND, "bucket the user named is gone"},
  };

  for (const auto& c : cases) {
    auto status = MakeExtendError(c.produced, "bad location", "bad location");

    EXPECT_EQ(ExternalSourceErrorCodeFromStatus(status, LOON_ARROW_ERROR), LOON_SOURCE_INVALID) << c.what;
    EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM) << c.what;

    // Everywhere else the producer's System verdict stands, because nothing
    // there knows whose string it was.
    EXPECT_EQ(FFIErrorCodeFromExtendStatus(status, LOON_ARROW_ERROR), c.internal_code) << c.what;
    EXPECT_EQ(loon_ffi_error_category(c.internal_code), LOON_ERROR_CATEGORY_SYSTEM) << c.what;
  }

  // The external-table property map is mixed: registered fs.* / writer.*
  // values are deployment configuration, while user extfs.* values are
  // validated later and arrive here as StorageConfigInvalid. A bare
  // InvalidProperties fallback therefore stays System.
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(arrow::Status::Invalid("x"), LOON_INVALID_PROPERTIES),
            LOON_INVALID_PROPERTIES);
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_PROPERTIES), LOON_ERROR_CATEGORY_SYSTEM);

  // A throttle remains retryable.
  auto throttled = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow", "slow");
  EXPECT_TRUE(loon_ffi_is_retryable_errcode(ExternalSourceErrorCodeFromStatus(throttled, LOON_ARROW_ERROR)));
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_SOURCE_INVALID));
}

// Requirement 4: the codes line up with the AWS S3 / Aliyun OSS vocabulary, so
// an operator can map a milvus-storage code to what the object store reported.
// Deliberate divergences are documented in docs/error-codes.md.
TEST(ErrorTaxonomyTest, S3VocabularyIsPinned) {
  auto s3_of = [](ExtendStatusCode code) { return std::string(S3CodeForExtendStatusCode(code)); };

  EXPECT_EQ(s3_of(ExtendStatusCode::StorageNotFound), "NoSuchKey");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageAccessDenied), "AccessDenied");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageNoSuchUpload), "NoSuchUpload");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageConflict), "OperationAborted");
  EXPECT_EQ(s3_of(ExtendStatusCode::StoragePreConditionFailed), "PreconditionFailed");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientThrottling), "SlowDown");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientService), "ServiceUnavailable");
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientTimeout), "RequestTimeout");
  EXPECT_EQ(s3_of(ExtendStatusCode::PackedInvalidArgs), "InvalidArgument");

  // Network transport failures, transaction failures and corruption have no
  // object-storage response code; an empty string is the documented way to
  // say so.
  EXPECT_EQ(s3_of(ExtendStatusCode::StorageTransientNetwork), "");
  EXPECT_EQ(s3_of(ExtendStatusCode::TxnExhaustedRetry), "");
  EXPECT_EQ(s3_of(ExtendStatusCode::TxnResolutionFailed), "");
  EXPECT_EQ(s3_of(ExtendStatusCode::PackedFileCorrupted), "");
}

// Documented divergences from AWS's own client/server split. Pinned so that
// changing one is a deliberate edit to both the table and docs/error-codes.md.
TEST(ErrorTaxonomyTest, DocumentedDivergencesFromAws) {
  // Conflict is a separate business signal, not permission for a generic retry
  // loop to replay the same conditional write.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageConflict), ErrorCategory::Conflict);
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StoragePreConditionFailed), ErrorCategory::Conflict);
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::TxnExhaustedRetry), ErrorCategory::Conflict);
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageConflict));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StoragePreConditionFailed));
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::TxnExhaustedRetry));
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_STORAGE_CONFLICT));

  // These codes carry the transient hint. Whether an enclosing operation may
  // be replayed is decided by the caller that owns that operation.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageTransientThrottling), ErrorCategory::Retryable);
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageTransientService), ErrorCategory::Retryable);

  // AWS: NoSuchKey is a 4xx client error. Ours: System -- on an internal path
  // the caller never chose the key, so it is not their fault. The specific code
  // lets milvus decide whether to re-read metadata or report missing data. The
  // external-source presentation counterpart is LOON_SOURCE_INVALID; both are
  // System because this layer does not validate or attribute user input.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageNotFound), ErrorCategory::System);
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageNotFound));

  // AWS: NoSuchBucket is grouped with NoSuchKey. We split it: nothing was lost,
  // and no re-read produces a bucket. It is a deployment pointing at something
  // that is not there, so it stays System and lands on BucketInvalid/2016.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageBucketNotFound), ErrorCategory::System);
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::StorageBucketNotFound), milvus::BucketInvalid);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::StorageBucketNotFound), milvus::ObjectNotExist);

  // AWS: NoSuchUpload is a 404 client error, and we used to call it Conflict and
  // retriable on the theory that our retry starts a fresh upload. That assumed a
  // consumer behaviour this layer cannot guarantee; a resend against the dead
  // upload id fails identically forever.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageNoSuchUpload), ErrorCategory::System);
  EXPECT_FALSE(RetryableForExtendStatusCode(ExtendStatusCode::StorageNoSuchUpload));
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM);

  // AWS: AccessDenied is a 403 client error. Ours: non-retryable System -- the
  // credentials are operator configuration, not part of the request.
  EXPECT_EQ(CategoryForExtendStatusCode(ExtendStatusCode::StorageAccessDenied), ErrorCategory::System);
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM);

  // AWS: InternalError (500) is retriable. Ours: LOON_LOGICAL_ERROR and friends
  // are our own bugs -- retrying reproduces them.
  EXPECT_EQ(loon_ffi_error_category(LOON_LOGICAL_ERROR), LOON_ERROR_CATEGORY_SYSTEM);
  EXPECT_EQ(loon_ffi_error_category(LOON_ARROW_ERROR), LOON_ERROR_CATEGORY_SYSTEM);

  // Storage failures remain System at every FFI entry point. A property value
  // that will not parse could equally be a milvus.yaml mistake or a segcore
  // hard-coded one, and this library does not own external-table validation.
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_PROPERTIES), LOON_ERROR_CATEGORY_SYSTEM);
  EXPECT_EQ(loon_ffi_error_category(LOON_STORAGE_CONFIG_INVALID), LOON_ERROR_CATEGORY_SYSTEM);
  EXPECT_EQ(loon_ffi_error_category(LOON_NOT_SUPPORT), LOON_ERROR_CATEGORY_SYSTEM);
  // User is minted only for a structurally valid FFI call whose caller-owned
  // API value violates that API's own contract.
  EXPECT_EQ(loon_ffi_error_category(LOON_SOURCE_INVALID), LOON_ERROR_CATEGORY_SYSTEM);
  EXPECT_EQ(loon_ffi_error_category(LOON_USER_INVALID_ARGUMENT), LOON_ERROR_CATEGORY_USER);
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_USER_INVALID_ARGUMENT));
  // Caller misuse across the C ABI is a developer's problem, not a user's.
  EXPECT_EQ(loon_ffi_error_category(LOON_INVALID_ARGS), LOON_ERROR_CATEGORY_SYSTEM);
}

// A defect in this library must not be reported as a storage incident. These
// three land on segcore's UnexpectedError so that whoever is paged reads
// "code bug", not "check the object store" -- the bucket they were in before
// is the same one a genuine S3 outage lands in.
TEST(ErrorTaxonomyTest, OurOwnBugsDoNotLookLikeStorageFailures) {
  const ExtendStatusCode our_bugs[] = {
      ExtendStatusCode::InternalInvariantViolated,
      ExtendStatusCode::PackedInvalidArgs,
  };
  for (auto code : our_bugs) {
    EXPECT_EQ(ToSegcoreErrorCode(code), milvus::UnexpectedError) << static_cast<int>(code);
    // System, never User: the caller of the C ABI did not cause it, and it is
    // not something a retry can resolve either.
    EXPECT_EQ(CategoryForExtendStatusCode(code), ErrorCategory::System) << static_cast<int>(code);
    EXPECT_FALSE(RetryableForExtendStatusCode(code)) << static_cast<int>(code);
  }

  // Real storage failures keep the storage bucket.
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::PackedIO), milvus::StorageError);

  const auto status = MakeExtendError(ExtendStatusCode::InternalInvariantViolated, "reader used after close");
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::UnexpectedError);
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(status, LOON_ARROW_ERROR), LOON_INTERNAL_INVARIANT);
}

// The int -> milvus::ErrorCode entry point must classify the internal FFI codes
// that never carry an ExtendStatusDetail: OOM, user input, deployment config,
// and our own bugs each get their own landing instead of collapsing into the
// generic StorageError.
TEST(ErrorTaxonomyTest, IntEntryPointClassifiesInternalCodes) {
  EXPECT_EQ(ToSegcoreErrorCode(LOON_MEMORY_ERROR), milvus::MemAllocateFailed);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_USER_INVALID_ARGUMENT), milvus::InvalidParameter);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_INVALID_PROPERTIES), milvus::ConfigInvalid);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_INVALID_ARGS), milvus::UnexpectedError);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_LOGICAL_ERROR), milvus::UnexpectedError);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_GOT_EXCEPTION), milvus::UnexpectedError);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_UNREACHABLE_ERROR), milvus::UnexpectedError);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_FILE_NOT_FOUND), milvus::ObjectNotExist);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_NOT_SUPPORT), milvus::ConfigInvalid);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_ARROW_ERROR), milvus::StorageError);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_SOURCE_INVALID), milvus::StorageError);

  // ExtendStatusCodes still delegate to the enum switch.
  EXPECT_EQ(ToSegcoreErrorCode(LOON_STORAGE_NOT_FOUND), milvus::ObjectNotExist);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_TRANSIENT_THROTTLING), milvus::StorageTransientError);

  // Unknown and retired values degrade to the conservative landing.
  EXPECT_EQ(ToSegcoreErrorCode(LOON_SUCCESS), milvus::StorageError);
  EXPECT_EQ(ToSegcoreErrorCode(9999), milvus::StorageError);
}

// A failed allocation reported through the FFI surface carries LOON_MEMORY_ERROR,
// not an internal-invariant code, so milvus sees OOM on both the direct-C++ and
// FFI paths.
TEST(ErrorTaxonomyTest, FfiOutOfMemoryUsesMemoryCode) {
  const auto oom = arrow::Status::OutOfMemory("allocation failed");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(oom, LOON_ARROW_ERROR), LOON_MEMORY_ERROR);
  EXPECT_EQ(ToSegcoreErrorCode(LOON_MEMORY_ERROR), milvus::MemAllocateFailed);
  EXPECT_EQ(ToSegcoreError(oom).get_error_code(), milvus::MemAllocateFailed);
}

// A dead multipart upload handle is a write-path fact, not "data missing": it
// must not share ObjectNotExist with StorageNotFound.
TEST(ErrorTaxonomyTest, NoSuchUploadIsNotObjectNotExist) {
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::StorageNoSuchUpload), milvus::StorageError);
  EXPECT_NE(ToSegcoreErrorCode(ExtendStatusCode::StorageNoSuchUpload), milvus::ObjectNotExist);
  EXPECT_EQ(ToSegcoreErrorCode(ExtendStatusCode::StorageNotFound), milvus::ObjectNotExist);
}

// Every bridge reports a missing object through the taxonomy channel (104).
// The errno channel (12) means the same thing and stays valid -- the
// filesystem layer and the vortex fork emit it -- but a consumer asking "is
// this a not-found?" must get the same answer from both, or the split becomes
// a second thing to remember.
TEST(ErrorTaxonomyTest, BothNotFoundChannelsAreEquivalent) {
  const auto taxonomy = MakeExtendError(ExtendStatusCode::StorageNotFound, "object is gone");
  const auto errno_based =
      arrow::Status::IOError("object is gone").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));

  EXPECT_EQ(ToSegcoreError(taxonomy).get_error_code(), milvus::ObjectNotExist);
  EXPECT_EQ(ToSegcoreError(errno_based).get_error_code(), milvus::ObjectNotExist);
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(taxonomy, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(errno_based, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
}

// NotImplemented has one meaning library-wide -- the capability is absent -- so
// it is mapped once, centrally, instead of at each entry point that happened to
// care. Invalid deliberately is NOT: it covers caller input, internal
// invariants and unparsable persisted bytes alike, so a central mapping would
// blame whoever the last call site guessed. It stays on the entry point's
// fallback until a producer classifies it.
TEST(ErrorTaxonomyTest, ArrowCodeIsMappedOnlyWhereItIsUnambiguous) {
  const auto not_implemented = arrow::Status::NotImplemented("equality deletes are not supported");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_implemented, LOON_ARROW_ERROR), LOON_NOT_SUPPORT);
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_implemented, LOON_INVALID_ARGS), LOON_NOT_SUPPORT);
  // The direct-C++ path reaches the same landing as the FFI path's code 9:
  // capability absence pages the deployment owner, not the object store.
  EXPECT_EQ(ToSegcoreError(not_implemented).get_error_code(), milvus::ConfigInvalid);

  const auto invalid = arrow::Status::Invalid("row group index out of range");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(invalid, LOON_ARROW_ERROR), LOON_ARROW_ERROR);
  EXPECT_EQ(loon_ffi_error_category(FFIErrorCodeFromExtendStatus(invalid, LOON_ARROW_ERROR)),
            LOON_ERROR_CATEGORY_SYSTEM);

  // A classified status still wins over both: the producer's verdict is never
  // overridden by the shape of the arrow code it travelled in.
  const auto corrupt = MakeExtendError(ExtendStatusCode::DataCorrupted, "manifest does not parse");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(corrupt, LOON_ARROW_ERROR), LOON_DATA_CORRUPTED);
}

// 2024 DataFormatBroken must come from a format-aware producer. The coarse
// arrow-status fallback used
// to guess it from Status::Invalid, which meant that of the ~380 unclassified
// Invalid sites in cpp/src -- null-pointer preconditions, absent configuration,
// caller contract violations -- every one arrived at segcore claiming a data
// format failure.
TEST(ErrorTaxonomyTest, CoarseFallbackNeverClaimsDataFormat) {
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
        << status.ToString() << " reached segcore claiming a data-format failure without a format-aware producer";
  }

  // ...while a producer that DID parse the bytes still gets there.
  for (auto format_error : {ExtendStatusCode::PackedMetadataCorrupted, ExtendStatusCode::PackedFileCorrupted,
                            ExtendStatusCode::DataCorrupted, ExtendStatusCode::VortexDataFormat}) {
    auto status = MakeExtendError(format_error, "bad bytes", "bad bytes");
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::DataFormatBroken);
  }

  // The ENOENT rung is untouched: a filesystem answering "no such file" is a
  // definitive not-found and keeps its specific ObjectNotExist mapping.
  auto enoent = arrow::Status::IOError("missing").WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  EXPECT_EQ(ToSegcoreError(enoent).get_error_code(), milvus::ObjectNotExist);
}

}  // namespace milvus_storage::test
