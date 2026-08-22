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

// Error categories crossing the C ABI. Codes retain detailed diagnosis; the
// category is a coarse fact/hint. It does not replace operation-aware policy.
//
//   USER         caller-owned input is invalid
//   RETRYABLE    the observed cause is transient; an operation-aware caller
//                decides whether and how to retry. A failed stateful writer
//                must be destroyed and recreated, never reused
//   CONFLICT     business-level coordination is required; the library does not
//                promise that replaying the same operation is safe or useful
//   DATA_FORMAT  persisted bytes do not decode
//   SYSTEM       every other non-user failure (configuration, missing internal
//                data, unsupported operation, bug, allocation failure, ...)
//
// UNKNOWN is consumer-side forward compatibility. No producer emits it, and it
// carries no generic retry hint; an operation owner may still apply its own
// idempotency/reconciliation policy.
#define LOON_ERROR_CATEGORY_UNKNOWN 0
#define LOON_ERROR_CATEGORY_USER 1
#define LOON_ERROR_CATEGORY_RETRYABLE 2
#define LOON_ERROR_CATEGORY_CONFLICT 3
#define LOON_ERROR_CATEGORY_DATA_FORMAT 4
#define LOON_ERROR_CATEGORY_SYSTEM 5

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
#define LOON_USER_INVALID_ARGUMENT 10
#define LOON_FILE_NOT_FOUND 12
// The external source cannot be resolved or accessed at this storage boundary.
// Deliberately ONE System code: S3
// answers a missing key with 403 rather than 404 when the caller lacks
// s3:ListBucket, specifically so object existence is not disclosed, so a
// not-found / access-denied split cannot be made accurately at this layer and
// would contradict the store's own security behaviour. External-table input
// validation belongs above this library. Which lower-layer condition occurred
// is retained in the message.
#define LOON_SOURCE_INVALID 13
// 14 retired (was LOON_SOURCE_ACCESS_DENIED, merged into 13). Never reuse.

// Shared with ExtendStatusCode. Keep values greater than arrow::StatusCode's current max value 45.
// 50-55 are reserved by Packed* ExtendStatusCode values.
#define LOON_STORAGE_NO_SUCH_UPLOAD 101
#define LOON_STORAGE_CONFLICT 102
#define LOON_STORAGE_PRECONDITION_FAILED 103
#define LOON_STORAGE_NOT_FOUND 104
#define LOON_STORAGE_ACCESS_DENIED 105
// 106 retired (was LOON_AWS_ERROR_NON_RETRYABLE). Never reuse. Deliberately
// absent from the tables below -- it has no category, no name and no producer
// -- while `loon_errcode_aws_non_retryable` stays exported as a tombstone so an
// older binding does not fail to load against a newer library (see ffi_c.h).
#define LOON_TRANSIENT_NETWORK 107
#define LOON_TRANSIENT_TIMEOUT 108
#define LOON_TRANSIENT_THROTTLING 109
#define LOON_TRANSIENT_SERVICE 110
#define LOON_TXN_EXHAUSTED_RETRY 111
#define LOON_TXN_RESOLUTION_FAILED 112
// 113-114 are reserved for the lance bridge codes introduced by #597.
// The storage location spec is unusable: the property map, the URI, or both.
// Deliberately ONE producer code. Low layers can receive deployment settings
// and external-source settings together, so they cannot reliably infer
// ownership from the field alone. They report System with this diagnostic code;
// an external-source entry point may coarsen its presentation to
// LOON_SOURCE_INVALID, which is also System. The precise field and reason
// belong in the message.
#define LOON_STORAGE_CONFIG_INVALID 115
// 116 retired (was LOON_SOURCE_URI_INVALID, merged into 115). Never reuse.
// Persisted bytes do not decode: the manifest, format metadata (paimon JSON,
// iceberg delete files, vortex footer), a LOB reference, a batch whose layout
// contradicts the schema. Deliberately ONE coarse code -- which artifact is
// broken belongs in the message, not in the code space. Distinct from the
// Packed* corruption codes only for diagnosis -- same category, same segcore
// landing.
#define LOON_DATA_CORRUPTED 117
// The bucket named by the deployment does not exist. NOT the same condition as
// a missing key: no amount of re-reading metadata produces a bucket, and no
// data was lost -- someone pointed the deployment at a bucket that is not
// there, which is a configuration fix.
#define LOON_STORAGE_BUCKET_NOT_FOUND 118
// A Vortex reader/decoder rejected the encoded data.
#define LOON_VORTEX_DATA_FORMAT 119
// 120 retired (was LOON_STORAGE_PARTIAL_FAILURE_RETRYABLE). Never reuse.
// 121 retired (was LOON_STORAGE_PARTIAL_FAILURE). Never reuse. Deliberately
// absent from the tables below -- no category, no name, no producer -- while
// `loon_errcode_storage_partial_failure_retryable` / `..._partial_failure`
// stay exported as tombstones so an older binding still loads (see ffi_c.h).
// Fan-out operations return their first concrete lower-layer failure instead.
// An internal invariant of this library was violated: a closed reader was
// reused, an index that the caller cannot influence was out of range, a branch
// documented as unreachable was reached. It is OUR bug, and it is the
// attachable counterpart of LOON_LOGICAL_ERROR -- which cannot be used here
// because ExtendStatusCode values must stay above arrow::StatusCode's range.
// It exists to keep code bugs out of the generic storage bucket: reported as a
// storage failure, they sent whoever was on call to inspect an object store
// that was working perfectly.
#define LOON_INTERNAL_INVARIANT 122

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
#define LOON_INTERNAL_ERROR_CODE_LIST(X)                                                                              \
  /* Caller violated the C ABI contract (null pointer, invalid handle, malformed C array). */                         \
  X("Invalid args", LOON_INVALID_ARGS, invalid_args, LOON_ERROR_CATEGORY_SYSTEM, "InvalidArgument")                   \
  /* Allocation failure. The one FFI code that expresses OOM: it maps to        */                     \
  /* segcore's MemAllocateFailed (2034) rather than being filed as a storage    */                     \
  /* incident. Non-retriable either way.                                        */                     \
  X("Memory allocation failure", LOON_MEMORY_ERROR, memory, LOON_ERROR_CATEGORY_SYSTEM, "")                           \
  /* Unclassified arrow failure. It carries no transient hint; the operation owner decides recovery. */               \
  X("Internal error", LOON_ARROW_ERROR, arrow, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")                           \
  /* Internal invariant violated -- our bug; retrying reproduces it. */                                               \
  X("Logical error", LOON_LOGICAL_ERROR, logical, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")                        \
  /* Catch-all for a C++ exception escaping an FFI entry point. */                                                    \
  X("Got exception", LOON_GOT_EXCEPTION, got_exception, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")                  \
  X("Unreachable code", LOON_UNREACHABLE_ERROR, unreachable, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")             \
  /* ConvertFFIProperties rejected a registered property's value (type, enum,                                         \
     range). These values arrive from deployment configuration (StorageConfig /                                       \
     milvus.yaml), which an operator can fix. Structural defects in the C property                                    \
     arrays are LOON_INVALID_ARGS instead, so this code has one owner. External-table                                 \
     calls carry the same fs.* configuration alongside user extfs.* properties;                                       \
     ConvertFFIProperties validates the former here, while the latter are classified                                  \
     later at the user-source boundary. */                                                                            \
  X("Invalid properties", LOON_INVALID_PROPERTIES, invalid_properties, LOON_ERROR_CATEGORY_SYSTEM, "InvalidArgument") \
  /* Test-only fault injection. */                                                                                    \
  X("Fault injection error", LOON_FAULT_INJECT_ERROR, fault_inject, LOON_ERROR_CATEGORY_SYSTEM, "")                   \
  /* The configured filesystem/reader lacks a capability this operation requires. */                                  \
  X("Not supported", LOON_NOT_SUPPORT, not_support, LOON_ERROR_CATEGORY_SYSTEM, "NotImplemented")                     \
  /* The FFI call is structurally valid, but a caller-owned API value violates its contract. */                       \
  X("User invalid argument", LOON_USER_INVALID_ARGUMENT, user_invalid_argument, LOON_ERROR_CATEGORY_USER,             \
    "InvalidArgument")                                                                                                \
  /* An object on an INTERNALLY generated path is missing: GC race, lost data, stale metadata. */                     \
  /* LOON_SOURCE_INVALID is the external-source-boundary counterpart. Both are System:        */                      \
  /* this library does not attribute source validation failures to a user.                     */                     \
  X("File not found", LOON_FILE_NOT_FOUND, file_not_found, LOON_ERROR_CATEGORY_SYSTEM, "NoSuchKey")                   \
  /* The external source is unusable: the path/bucket does not exist, or access to it was       */                    \
  /* denied. Same object-store conditions as                                                     */                   \
  /* LOON_FILE_NOT_FOUND and StorageAccessDenied at the external-source boundary. ONE code: S3 */                    \
  /* answers a missing key with 403 rather than 404 when the caller lacks s3:ListBucket, so the */                    \
  /* split cannot be made accurately here. Only entry points contractually handed a             */                    \
  /* external-source entry points mint this -- see ExternalSourceErrorCodeFromStatus().         */                    \
  X("Source invalid", LOON_SOURCE_INVALID, source_invalid, LOON_ERROR_CATEGORY_SYSTEM, "NoSuchBucket")

// Codes shared with ExtendStatusCode: these can be attached to an arrow::Status
// as an ExtendStatusDetail and survive from the producing layer all the way to
// the FFI / segcore boundary.
#define LOON_EXTEND_STATUS_CODE_LIST(X)                                                                                \
  /* --- packed layer (50-55) --- */                                                                                   \
  X(PackedInvalidArgs, 50, packed_invalid_args, LOON_ERROR_CATEGORY_SYSTEM, "InvalidArgument")                         \
  X(PackedIO, 51, packed_io, LOON_ERROR_CATEGORY_SYSTEM, "")                                                           \
  X(PackedMetadataCorrupted, 52, packed_metadata_corrupted, LOON_ERROR_CATEGORY_DATA_FORMAT, "")                       \
  X(PackedFileCorrupted, 53, packed_file_corrupted, LOON_ERROR_CATEGORY_DATA_FORMAT, "")                               \
  /* 54 retired (was PackedIOTransient). Never reuse. Its symbol                 */                                   \
  /* loon_errcode_packed_io_transient stays exported as a tombstone (see ffi_c.h);*/                                   \
  /* dependency I/O errors are preserved instead of being broadly reclassified.  */                                   \
  X(PackedUnexpected, 55, packed_unexpected, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")                              \
  /* --- object storage (101-110) --- */                                                                               \
  /* The multipart upload the caller held a handle to is gone. It is not Conflict: the only    */                      \
  /* thing that helps is starting a NEW upload, and resending against the dead upload id fails  */                     \
  /* identically every time. Marking it retriable invited exactly that blind resend -- the      */                     \
  /* earlier justification ("our retry is at the operation level and creates a fresh upload")   */                     \
  /* assumed a consumer behaviour this layer cannot guarantee. The layer that owns the write    */                     \
  /* decides whether to redo it; we only report that the handle is dead.                        */                     \
  X(StorageNoSuchUpload, LOON_STORAGE_NO_SUCH_UPLOAD, storage_no_such_upload, LOON_ERROR_CATEGORY_SYSTEM,               \
    "NoSuchUpload")                                                                                                    \
  X(StorageConflict, LOON_STORAGE_CONFLICT, storage_conflict, LOON_ERROR_CATEGORY_CONFLICT, "OperationAborted")         \
  X(StoragePreConditionFailed, LOON_STORAGE_PRECONDITION_FAILED, storage_precondition_failed,                           \
    LOON_ERROR_CATEGORY_CONFLICT, "PreconditionFailed")                                                                \
  /* NoSuchKey / ResourceNotFound on an internally generated path. This layer records the       */                     \
  /* absence but leaves any manifest re-read / recovery decision to its consumer.               */                     \
  X(StorageNotFound, LOON_STORAGE_NOT_FOUND, storage_not_found, LOON_ERROR_CATEGORY_SYSTEM, "NoSuchKey")                \
  /* AccessDenied / InvalidAccessKeyId / SignatureDoesNotMatch. These credentials are operator  */                     \
  /* configuration, so the caller did not cause the failure and generic retry cannot fix it.    */                     \
  /* It stays System and should reach whoever owns the deployment. An external-source entry     */                     \
  /* point may report the unified System code LOON_SOURCE_INVALID instead.                      */                     \
  X(StorageAccessDenied, LOON_STORAGE_ACCESS_DENIED, storage_access_denied, LOON_ERROR_CATEGORY_SYSTEM, "AccessDenied") \
  X(StorageTransientNetwork, LOON_TRANSIENT_NETWORK, transient_network, LOON_ERROR_CATEGORY_RETRYABLE, "")             \
  X(StorageTransientTimeout, LOON_TRANSIENT_TIMEOUT, transient_timeout, LOON_ERROR_CATEGORY_RETRYABLE,                 \
    "RequestTimeout")                                                                                                  \
  X(StorageTransientThrottling, LOON_TRANSIENT_THROTTLING, transient_throttling, LOON_ERROR_CATEGORY_RETRYABLE,        \
    "SlowDown")                                                                                                        \
  X(StorageTransientService, LOON_TRANSIENT_SERVICE, transient_service, LOON_ERROR_CATEGORY_RETRYABLE,                 \
    "ServiceUnavailable")                                                                                              \
  /* --- manifest transactions (111-112): no object-storage counterpart --- */                                         \
  /* The transaction's own retry budget is spent; repeating the same commit will not help. */                          \
  X(TxnExhaustedRetry, LOON_TXN_EXHAUSTED_RETRY, txn_exhausted_retry, LOON_ERROR_CATEGORY_CONFLICT, "")                \
  X(TxnResolutionFailed, LOON_TXN_RESOLUTION_FAILED, txn_resolution_failed, LOON_ERROR_CATEGORY_CONFLICT, "")          \
  /* --- config and persisted-data classification (115-119) --- */                                                     \
  /* Storage configuration the operator supplied is unusable: unknown cloud   */                                       \
  /* provider, malformed extfs.* property, endpoint/credential fields that do */                                       \
  /* not parse. Nobody can retry their way out of it.                          */                                      \
  X(StorageConfigInvalid, LOON_STORAGE_CONFIG_INVALID, storage_config_invalid, LOON_ERROR_CATEGORY_SYSTEM,             \
    "InvalidArgument")                                                                                                 \
  /* The manifest does not parse: bad MILV magic, malformed avro body. Its own  */                                     \
  /* code purely for diagnosis -- DataFormat, 2024, same as the Packed* pair.   */                                     \
  X(DataCorrupted, LOON_DATA_CORRUPTED, data_corrupted, LOON_ERROR_CATEGORY_DATA_FORMAT, "")                           \
  /* NoSuchBucket, split out of StorageNotFound. A missing bucket on an        */                     \
  /* internally generated path is a deployment pointing somewhere that does not */                     \
  /* exist, not data loss: re-reading the manifest cannot conjure a bucket.     */                     \
  X(StorageBucketNotFound, LOON_STORAGE_BUCKET_NOT_FOUND, storage_bucket_not_found, LOON_ERROR_CATEGORY_SYSTEM,         \
    "NoSuchBucket")                                                                                                    \
  /* A Vortex decoder rejected file metadata or encoded column data. This is a  */                                     \
  /* broad format error, not a quarantine/rebuild verdict. Transport markers    */                                     \
  /* and caught panics keep their own non-format classification.                 */                                    \
  X(VortexDataFormat, LOON_VORTEX_DATA_FORMAT, vortex_data_format, LOON_ERROR_CATEGORY_DATA_FORMAT, "")                \
  /* 120/121 retired (was StoragePartialFailureRetryable / StoragePartialFailure). */                                 \
  /* Never reuse -- see the #define comments above. Their symbols stay exported as     */                             \
  /* tombstones. Fan-out operations return their first concrete lower-layer failure.    */                             \
  /* --- our own bugs (122) --- */                                                                                     \
  /* An invariant of this library was violated: a closed reader reused, an index the caller     */                     \
  /* cannot influence out of range, an unreachable branch reached. System like every other      */                     \
  /* non-user failure, but it lands on segcore's UnexpectedError instead of StorageError, so a  */                     \
  /* code defect stops being reported as a storage incident.                                     */                    \
  X(InternalInvariantViolated, LOON_INTERNAL_INVARIANT, internal_invariant, LOON_ERROR_CATEGORY_SYSTEM, "InternalError")
