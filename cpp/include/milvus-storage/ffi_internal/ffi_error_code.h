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

#pragma once

// ===========================================================================
// Error categories -- the closed set that crosses the C ABI.
//
// Every error milvus-storage hands to an upper layer must answer two
// questions, and BOTH are derived from this one value:
//
//   * whose problem is it -- the caller's request/config (User), or ours
//     (Transient / Permanent)?
//   * can a retry possibly help -- only Transient.
//
// Invariant (enforced by error_taxonomy_test.cpp): retryable == (category ==
// Transient). A single `retryable` bool is deliberately NOT the primary field:
// User and Permanent are both non-retriable but need opposite handling --
// report a User error back to whoever made the request, alert an operator for
// a Permanent one.
//
// The category is a pure function of the error code, so it needs no room in
// LoonFFIResult and a consumer can classify a code it has never seen.
// ===========================================================================
#define LOON_ERROR_CATEGORY_UNKNOWN 0
#define LOON_ERROR_CATEGORY_USER 1
#define LOON_ERROR_CATEGORY_TRANSIENT 2
#define LOON_ERROR_CATEGORY_PERMANENT 3

#define LOON_SUCCESS 0
#define LOON_INVALID_ARGS 1
#define LOON_MEMORY_ERROR 2
#define LOON_ARROW_ERROR 3
#define LOON_LOGICAL_ERROR 4
#define LOON_GOT_EXCEPTION 5
#define LOON_UNREACHABLE_ERROR 6
#define LOON_INVALID_PROPERTIES 7
#define LOON_FAULT_INJECT_ERROR 8
#define LOON_NOT_SUPPORT 9
#define LOON_FILE_NOT_FOUND 12
#define LOON_SOURCE_NOT_FOUND 13
#define LOON_SOURCE_ACCESS_DENIED 14

// Shared with ExtendStatusCode. Keep values greater than arrow::StatusCode's current max value 45.
// 50-55 are reserved by Packed* ExtendStatusCode values.
#define LOON_AWS_ERROR_NO_SUCH_UPLOAD 101
#define LOON_AWS_ERROR_CONFLICT 102
#define LOON_AWS_ERROR_PRECONDITION_FAILED 103
#define LOON_AWS_ERROR_NOT_FOUND 104
#define LOON_AWS_ERROR_ACCESS_DENIED 105
#define LOON_AWS_ERROR_NON_RETRYABLE 106
#define LOON_TRANSIENT_NETWORK 107
#define LOON_TRANSIENT_TIMEOUT 108
#define LOON_TRANSIENT_THROTTLING 109
#define LOON_TRANSIENT_SERVICE 110
#define LOON_TXN_EXHAUSTED_RETRY 111
#define LOON_TXN_RESOLUTION_FAILED 112

// ===========================================================================
// THE error tables. Everything downstream -- the exported `loon_errcode_*`
// constants, `error_to_string`, `loon_ffi_error_category`,
// `loon_ffi_is_retryable_errcode`, `enum class ExtendStatusCode` and its
// metadata -- is generated from the two lists below, so a code cannot be
// classified two different ways in two different tables.
//
// Adding a code here is the only edit needed; forgetting to classify it fails
// the build (`ToSegcoreErrorCode` is a no-`default` switch under
// `-Werror=switch`) or the taxonomy test.
//
// Columns:
//   name      human-readable name; also the metrics/log label
//   code      numeric value -- this is the C ABI contract, never renumber
//   symbol    suffix of the exported constant `loon_errcode_<symbol>`
//   category  LOON_ERROR_CATEGORY_* -- see the invariant above
//   s3_code   the AWS S3 / Aliyun OSS error code this corresponds to, or ""
//             when the condition has no object-storage counterpart.
//             docs/error-codes.md documents every deliberate divergence.
// ===========================================================================

// Codes minted by the FFI layer itself (argument checks, catch-all handlers).
// They never carry an ExtendStatusDetail, so they never appear as an
// ExtendStatusCode.
#define LOON_INTERNAL_ERROR_CODE_LIST(X)                                                                            \
  /* Caller passed bad arguments across the C ABI (null pointer, empty path, bad range). */                         \
  X("Invalid args", LOON_INVALID_ARGS, invalid_args, LOON_ERROR_CATEGORY_USER, "InvalidArgument")                   \
  /* Local allocation failed. Retriable: another node, or this one later, may have the memory. */                   \
  X("Memory allocation failed", LOON_MEMORY_ERROR, memory, LOON_ERROR_CATEGORY_TRANSIENT, "")                       \
  /* Unclassified arrow failure. Conservative: an unknown failure is never retried. */                              \
  X("Internal error", LOON_ARROW_ERROR, arrow, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                      \
  /* Internal invariant violated -- our bug; retrying reproduces it. */                                             \
  X("Logical error", LOON_LOGICAL_ERROR, logical, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                   \
  /* Catch-all for a C++ exception escaping an FFI entry point. */                                                  \
  X("Got exception", LOON_GOT_EXCEPTION, got_exception, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")             \
  X("Unreachable code", LOON_UNREACHABLE_ERROR, unreachable, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")        \
  /* The properties the caller supplied are malformed or unusable. */                                               \
  X("Invalid properties", LOON_INVALID_PROPERTIES, invalid_properties, LOON_ERROR_CATEGORY_USER, "InvalidArgument") \
  /* Test-only fault injection. */                                                                                  \
  X("Fault injection error", LOON_FAULT_INJECT_ERROR, fault_inject, LOON_ERROR_CATEGORY_PERMANENT, "")              \
  /* The caller asked for a feature/format this build does not implement. */                                        \
  X("Not supported", LOON_NOT_SUPPORT, not_support, LOON_ERROR_CATEGORY_USER, "NotImplemented")                     \
  /* An object on an INTERNALLY generated path is missing: GC race, lost data, stale metadata. */                   \
  /* Not a User error -- the caller never chose this path. LOON_SOURCE_NOT_FOUND is the        */                   \
  /* user-supplied-path counterpart.                                                           */                   \
  X("File not found", LOON_FILE_NOT_FOUND, file_not_found, LOON_ERROR_CATEGORY_PERMANENT, "NoSuchKey")              \
  /* A path/bucket the USER named (an external-source URI) does not exist: same object-store   */                   \
  /* condition as LOON_FILE_NOT_FOUND, opposite ownership. Only entry points that take a       */                   \
  /* user-supplied location mint this -- see UserSourceErrorCodeFromStatus().                  */                   \
  X("Source not found", LOON_SOURCE_NOT_FOUND, source_not_found, LOON_ERROR_CATEGORY_USER, "NoSuchBucket")          \
  /* The credentials the USER supplied for an external source were rejected. */                                     \
  X("Source access denied", LOON_SOURCE_ACCESS_DENIED, source_access_denied, LOON_ERROR_CATEGORY_USER, "AccessDenied")

// Codes shared with ExtendStatusCode: these can be attached to an arrow::Status
// as an ExtendStatusDetail and survive from the producing layer all the way to
// the FFI / segcore boundary.
#define LOON_EXTEND_STATUS_CODE_LIST(X)                                                                         \
  /* --- packed layer (50-55) --- */                                                                            \
  X(PackedInvalidArgs, 50, packed_invalid_args, LOON_ERROR_CATEGORY_USER, "InvalidArgument")                    \
  X(PackedStorageIO, 51, packed_storage_io, LOON_ERROR_CATEGORY_PERMANENT, "")                                  \
  X(PackedMetadataCorrupted, 52, packed_metadata_corrupted, LOON_ERROR_CATEGORY_PERMANENT, "")                  \
  X(PackedFileCorrupted, 53, packed_file_corrupted, LOON_ERROR_CATEGORY_PERMANENT, "")                          \
  X(PackedArrowError, 54, packed_arrow_error, LOON_ERROR_CATEGORY_PERMANENT, "")                                \
  X(PackedUnexpected, 55, packed_unexpected, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                    \
  /* --- object storage (101-110) --- */                                                                        \
  /* The multipart upload state is gone. AWS calls NoSuchUpload a 404 client error; we classify */              \
  /* it Transient because our retry is at the operation level and creates a fresh upload.       */              \
  X(AwsErrorNoSuchUpload, LOON_AWS_ERROR_NO_SUCH_UPLOAD, aws_no_such_upload, LOON_ERROR_CATEGORY_TRANSIENT,     \
    "NoSuchUpload")                                                                                             \
  X(AwsErrorConflict, LOON_AWS_ERROR_CONFLICT, aws_conflict, LOON_ERROR_CATEGORY_PERMANENT, "OperationAborted") \
  X(AwsErrorPreConditionFailed, LOON_AWS_ERROR_PRECONDITION_FAILED, aws_precondition_failed,                    \
    LOON_ERROR_CATEGORY_PERMANENT, "PreconditionFailed")                                                        \
  /* NoSuchKey / NoSuchBucket / ResourceNotFound on an internally generated path. Permanent: a  */              \
  /* retry, or a reroute to another replica, hits the same shared object store.                 */              \
  X(AwsErrorNotFound, LOON_AWS_ERROR_NOT_FOUND, aws_not_found, LOON_ERROR_CATEGORY_PERMANENT, "NoSuchKey")      \
  /* AccessDenied / InvalidAccessKeyId / SignatureDoesNotMatch. Permanent, not User: these      */              \
  /* credentials are operator configuration, not part of the caller's request.                  */              \
  X(AwsErrorAccessDenied, LOON_AWS_ERROR_ACCESS_DENIED, aws_access_denied, LOON_ERROR_CATEGORY_PERMANENT,       \
    "AccessDenied")                                                                                             \
  /* Any other error the AWS SDK itself judged non-retryable (ShouldRetry() == false). */                       \
  X(AwsErrorNonRetryable, LOON_AWS_ERROR_NON_RETRYABLE, aws_non_retryable, LOON_ERROR_CATEGORY_PERMANENT, "")   \
  X(StorageTransientNetwork, LOON_TRANSIENT_NETWORK, transient_network, LOON_ERROR_CATEGORY_TRANSIENT,          \
    "RequestTimeout")                                                                                           \
  X(StorageTransientTimeout, LOON_TRANSIENT_TIMEOUT, transient_timeout, LOON_ERROR_CATEGORY_TRANSIENT,          \
    "RequestTimeout")                                                                                           \
  X(StorageTransientThrottling, LOON_TRANSIENT_THROTTLING, transient_throttling, LOON_ERROR_CATEGORY_TRANSIENT, \
    "SlowDown")                                                                                                 \
  X(StorageTransientService, LOON_TRANSIENT_SERVICE, transient_service, LOON_ERROR_CATEGORY_TRANSIENT,          \
    "ServiceUnavailable")                                                                                       \
  /* --- manifest transactions (111-112): no object-storage counterpart --- */                                  \
  /* The transaction's own retry budget is spent; repeating the same commit will not help. */                   \
  X(TxnExhaustedRetry, LOON_TXN_EXHAUSTED_RETRY, txn_exhausted_retry, LOON_ERROR_CATEGORY_PERMANENT, "")        \
  X(TxnResolutionFailed, LOON_TXN_RESOLUTION_FAILED, txn_resolution_failed, LOON_ERROR_CATEGORY_PERMANENT, "")
