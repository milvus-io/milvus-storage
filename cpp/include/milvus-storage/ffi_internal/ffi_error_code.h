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
// Seven categories, chosen so that each maps to exactly ONE consumer action.
// That is the test for whether a category earns its place: if two conditions
// call for the same action they belong together, and a category that names an
// action the code cannot actually supply does not belong at all.
//
//   USER       the user's own request is wrong   -> return it to them,
//                                                   do not alert, never retry
//   CONFIG     the deployment is wrong, or we    -> alert an operator,
//              cannot tell whose input it was       never retry
//   TRANSIENT  a condition that may clear        -> back off, retry the
//                                                   same request
//   CONFLICT   someone else won a race           -> re-read state, rebase,
//                                                   re-submit. NOT a replay of
//                                                   the same bytes, which
//                                                   fails identically
//   MISSING    the named object is not there     -> re-read the metadata, then
//                                                   decide. NOT retried here:
//                                                   see below
//   CORRUPTED  the bytes are wrong               -> act on the DATA: quarantine,
//                                                   re-fetch from a replica,
//                                                   rebuild
//   PERMANENT  our bug, or the data is gone      -> alert a developer,
//                                                   never retry
//
// Retriability is derived, never stored:
//   retryable == (TRANSIENT || CONFLICT)
//
// MISSING is NOT retriable, and that is a deliberate refusal rather than a
// verdict. A missing object on an internally generated path is either a GC race
// (the file was legitimately collected; re-reading the manifest finds it gone
// and the caller moves on) or real data loss (an operator must look). This
// layer cannot tell those apart -- and inventing retriability is the one thing
// this taxonomy never does. milvus CAN tell, by re-reading the manifest, so the
// retry decision belongs there. The category exists to say "re-read before you
// decide", which is neither Transient's "send it again" nor Permanent's "give
// up".
//
// MISSING and CORRUPTED each carry a discipline that is machine-checkable:
// only a producer that got a DEFINITIVE not-found from the store may say
// MISSING, and only a producer that actually PARSED the bytes and found them
// wrong may say CORRUPTED. Neither may ever be inferred from an unclassified
// failure -- which is why the coarse arrow-status fallback in ToSegcoreError
// deliberately lands on StorageError rather than guessing DataFormatBroken.
//
// USER is deliberately narrow. Only an entry point that is contractually
// handed a user-supplied location can know the input came from a user, so USER
// is produced exclusively by the re-tagging at those entry points -- no layer
// below may classify itself USER. Everything we cannot attribute goes to
// CONFIG: telling an operator to check configuration for what turns out to be
// a user typo wastes their time, while telling a user "your parameter is bad"
// for a broken deployment sends them editing their query forever and nobody
// ever looks at the real cause.
//
// There is no THROTTLED. Rate limiting is still identified as its own CODE
// (LOON_TRANSIENT_THROTTLING) so it can be logged and measured separately, but
// it is not a separate CATEGORY: the action that would distinguish it is "back
// off for the duration the store told us", and we never extract Retry-After --
// LoonFFIResult has no channel to carry a per-occurrence duration. A category
// promising an action the consumer cannot perform is worse than no category.
// Add it back together with that channel, not before.
//
// UNKNOWN is not an eighth kind of error -- no producer ever emits it. It is what
// a CONSUMER reports for a code newer than itself, and it must be handled as
// PERMANENT: never retry what you cannot classify.
//
// The category is a pure function of the error code, so it needs no room in
// LoonFFIResult and a consumer can classify a code it has never seen.
// ===========================================================================
#define LOON_ERROR_CATEGORY_UNKNOWN 0
#define LOON_ERROR_CATEGORY_USER 1
#define LOON_ERROR_CATEGORY_CONFIG 2
#define LOON_ERROR_CATEGORY_TRANSIENT 3
#define LOON_ERROR_CATEGORY_CONFLICT 4
#define LOON_ERROR_CATEGORY_MISSING 5
#define LOON_ERROR_CATEGORY_CORRUPTED 6
#define LOON_ERROR_CATEGORY_PERMANENT 7

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
// The external source the user named is unusable -- it does not exist, or the
// credentials they supplied for it were rejected. Deliberately ONE code: S3
// answers a missing key with 403 rather than 404 when the caller lacks
// s3:ListBucket, specifically so object existence is not disclosed, so a
// not-found / access-denied split cannot be made accurately at this layer and
// would contradict the store's own security behaviour. Which one it was is in
// the message.
#define LOON_SOURCE_INVALID 13
// 14 retired (was LOON_SOURCE_ACCESS_DENIED, merged into 13). Never reuse.

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
// 113-114 are reserved for the lance bridge codes introduced by #597.
// The storage location spec is unusable: the property map, the URI, or both.
// Deliberately ONE producer code. Low layers can receive deployment settings
// and user-source settings together, so they cannot reliably infer ownership
// from the field alone. They report Config here; an entry point that knows the
// failing operation was resolving a user-supplied location may re-tag it to
// LOON_SOURCE_INVALID. The precise field and reason belong in the message.
#define LOON_STORAGE_CONFIG_INVALID 115
// 116 retired (was LOON_SOURCE_URI_INVALID, merged into 115). Never reuse.
// The manifest itself does not parse. Distinct from the Packed* corruption
// codes only for diagnosis -- same category, same segcore landing.
#define LOON_MANIFEST_CORRUPTED 117
// The bucket named by the deployment does not exist. NOT the same condition as
// a missing key: no amount of re-reading metadata produces a bucket, and no
// data was lost -- someone pointed the deployment at a bucket that is not
// there, which is a configuration fix.
#define LOON_AWS_ERROR_BUCKET_NOT_FOUND 118
// A vortex file does not parse. Same category and landing as 117; separate so
// a corruption alert says which format failed without parsing the message.
#define LOON_VORTEX_FILE_CORRUPTED 119

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
  /* Caller passed bad arguments across the C ABI (null pointer, empty path, bad range). */                           \
  X("Invalid args", LOON_INVALID_ARGS, invalid_args, LOON_ERROR_CATEGORY_PERMANENT, "InvalidArgument")                \
  /* Local allocation failed. Retriable: another node, or this one later, may have the memory. */                     \
  X("Memory allocation failed", LOON_MEMORY_ERROR, memory, LOON_ERROR_CATEGORY_TRANSIENT, "")                         \
  /* Unclassified arrow failure. Conservative: an unknown failure is never retried. */                                \
  X("Internal error", LOON_ARROW_ERROR, arrow, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                        \
  /* Internal invariant violated -- our bug; retrying reproduces it. */                                               \
  X("Logical error", LOON_LOGICAL_ERROR, logical, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                     \
  /* Catch-all for a C++ exception escaping an FFI entry point. */                                                    \
  X("Got exception", LOON_GOT_EXCEPTION, got_exception, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")               \
  X("Unreachable code", LOON_UNREACHABLE_ERROR, unreachable, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")          \
  /* Two producer families share this code. (1) loon_properties_create's                                              \
     structural checks -- duplicate key, null key or value: the FFI caller's                                          \
     marshalling bug, vanishingly rare. (2) ConvertFFIProperties rejecting a                                          \
     registered property whose value fails its validator (type, enum, range):                                         \
     those values arrive from deployment configuration (StorageConfig /                                               \
     milvus.yaml), which an operator can actually fix. External-table calls                                           \
     carry the same fs.* configuration alongside user extfs.* properties;                                             \
     ConvertFFIProperties validates the former here, while the latter are                                             \
     classified later at the user-source boundary. */                                                                 \
  X("Invalid properties", LOON_INVALID_PROPERTIES, invalid_properties, LOON_ERROR_CATEGORY_CONFIG, "InvalidArgument") \
  /* Test-only fault injection. */                                                                                    \
  X("Fault injection error", LOON_FAULT_INJECT_ERROR, fault_inject, LOON_ERROR_CATEGORY_PERMANENT, "")                \
  /* The caller asked for a feature/format this build does not implement. */                                          \
  X("Not supported", LOON_NOT_SUPPORT, not_support, LOON_ERROR_CATEGORY_CONFIG, "NotImplemented")                     \
  /* An object on an INTERNALLY generated path is missing: GC race, lost data, stale metadata. */                     \
  /* Not a User error -- the caller never chose this path. LOON_SOURCE_INVALID is the          */                     \
  /* user-supplied-path counterpart.                                                           */                     \
  X("File not found", LOON_FILE_NOT_FOUND, file_not_found, LOON_ERROR_CATEGORY_MISSING, "NoSuchKey")                  \
  /* The external source the USER named is unusable: the path/bucket does not exist, or the     */                    \
  /* credentials they supplied for it were rejected. Same object-store conditions as            */                    \
  /* LOON_FILE_NOT_FOUND and AwsErrorAccessDenied, opposite ownership. ONE code on purpose: S3  */                    \
  /* answers a missing key with 403 rather than 404 when the caller lacks s3:ListBucket, so the */                    \
  /* split cannot be made accurately here. Only entry points contractually handed a             */                    \
  /* user-supplied location mint this -- see UserSourceErrorCodeFromStatus().                   */                    \
  X("Source invalid", LOON_SOURCE_INVALID, source_invalid, LOON_ERROR_CATEGORY_USER, "NoSuchBucket")

// Codes shared with ExtendStatusCode: these can be attached to an arrow::Status
// as an ExtendStatusDetail and survive from the producing layer all the way to
// the FFI / segcore boundary.
#define LOON_EXTEND_STATUS_CODE_LIST(X)                                                                                \
  /* --- packed layer (50-55) --- */                                                                                   \
  X(PackedInvalidArgs, 50, packed_invalid_args, LOON_ERROR_CATEGORY_PERMANENT, "InvalidArgument")                      \
  X(PackedStorageIO, 51, packed_storage_io, LOON_ERROR_CATEGORY_PERMANENT, "")                                         \
  X(PackedMetadataCorrupted, 52, packed_metadata_corrupted, LOON_ERROR_CATEGORY_CORRUPTED, "")                         \
  X(PackedFileCorrupted, 53, packed_file_corrupted, LOON_ERROR_CATEGORY_CORRUPTED, "")                                 \
  X(PackedArrowError, 54, packed_arrow_error, LOON_ERROR_CATEGORY_PERMANENT, "")                                       \
  X(PackedUnexpected, 55, packed_unexpected, LOON_ERROR_CATEGORY_PERMANENT, "InternalError")                           \
  /* --- object storage (101-110) --- */                                                                               \
  /* The multipart upload the caller held a handle to is gone. Missing, not Conflict: the only  */                     \
  /* thing that helps is starting a NEW upload, and resending against the dead upload id fails  */                     \
  /* identically every time. Marking it retriable invited exactly that blind resend -- the      */                     \
  /* earlier justification ("our retry is at the operation level and creates a fresh upload")   */                     \
  /* assumed a consumer behaviour this layer cannot guarantee. The layer that owns the write    */                     \
  /* decides whether to redo it; we only report that the handle is dead.                        */                     \
  X(AwsErrorNoSuchUpload, LOON_AWS_ERROR_NO_SUCH_UPLOAD, aws_no_such_upload, LOON_ERROR_CATEGORY_MISSING,              \
    "NoSuchUpload")                                                                                                    \
  X(AwsErrorConflict, LOON_AWS_ERROR_CONFLICT, aws_conflict, LOON_ERROR_CATEGORY_CONFLICT, "OperationAborted")         \
  X(AwsErrorPreConditionFailed, LOON_AWS_ERROR_PRECONDITION_FAILED, aws_precondition_failed,                           \
    LOON_ERROR_CATEGORY_CONFLICT, "PreconditionFailed")                                                                \
  /* NoSuchKey / ResourceNotFound on an internally generated path. Missing: this layer          */                     \
  /* records the absence but leaves any manifest re-read / retry decision to its consumer.      */                     \
  X(AwsErrorNotFound, LOON_AWS_ERROR_NOT_FOUND, aws_not_found, LOON_ERROR_CATEGORY_MISSING, "NoSuchKey")               \
  /* AccessDenied / InvalidAccessKeyId / SignatureDoesNotMatch. Config, not User and not        */                     \
  /* Permanent: these credentials are operator configuration, so the caller did not cause it    */                     \
  /* and no retry escapes it -- it has to page whoever owns the deployment. When the            */                     \
  /* credentials came from the user instead, an external-source entry point re-tags this to     */                     \
  /* LOON_SOURCE_INVALID, which is User.                                                        */                     \
  X(AwsErrorAccessDenied, LOON_AWS_ERROR_ACCESS_DENIED, aws_access_denied, LOON_ERROR_CATEGORY_CONFIG, "AccessDenied") \
  /* Any other error the AWS SDK itself judged non-retryable (ShouldRetry() == false). */                              \
  X(AwsErrorNonRetryable, LOON_AWS_ERROR_NON_RETRYABLE, aws_non_retryable, LOON_ERROR_CATEGORY_PERMANENT, "")          \
  X(StorageTransientNetwork, LOON_TRANSIENT_NETWORK, transient_network, LOON_ERROR_CATEGORY_TRANSIENT, "")             \
  X(StorageTransientTimeout, LOON_TRANSIENT_TIMEOUT, transient_timeout, LOON_ERROR_CATEGORY_TRANSIENT,                 \
    "RequestTimeout")                                                                                                  \
  X(StorageTransientThrottling, LOON_TRANSIENT_THROTTLING, transient_throttling, LOON_ERROR_CATEGORY_TRANSIENT,        \
    "SlowDown")                                                                                                        \
  X(StorageTransientService, LOON_TRANSIENT_SERVICE, transient_service, LOON_ERROR_CATEGORY_TRANSIENT,                 \
    "ServiceUnavailable")                                                                                              \
  /* --- manifest transactions (111-112): no object-storage counterpart --- */                                         \
  /* The transaction's own retry budget is spent; repeating the same commit will not help. */                          \
  X(TxnExhaustedRetry, LOON_TXN_EXHAUSTED_RETRY, txn_exhausted_retry, LOON_ERROR_CATEGORY_CONFLICT, "")                \
  X(TxnResolutionFailed, LOON_TXN_RESOLUTION_FAILED, txn_resolution_failed, LOON_ERROR_CATEGORY_CONFLICT, "")          \
  /* --- config / caller-supplied location (115-116) --- */                                                            \
  /* Storage configuration the operator supplied is unusable: unknown cloud   */                                       \
  /* provider, malformed extfs.* property, endpoint/credential fields that do */                                       \
  /* not parse. Nobody can retry their way out of it.                          */                                      \
  X(StorageConfigInvalid, LOON_STORAGE_CONFIG_INVALID, storage_config_invalid, LOON_ERROR_CATEGORY_CONFIG,             \
    "InvalidArgument")                                                                                                 \
  /* The manifest does not parse: bad MILV magic, malformed avro body. Its own  */                                     \
  /* code purely for diagnosis -- Corrupted, 2024, same as the Packed* pair.    */                                     \
  X(ManifestCorrupted, LOON_MANIFEST_CORRUPTED, manifest_corrupted, LOON_ERROR_CATEGORY_CORRUPTED, "")                 \
  /* NoSuchBucket, split out of AwsErrorNotFound. A missing bucket on an        */                                     \
  /* internally generated path is a deployment pointing somewhere that does not */                                     \
  /* exist, not data loss: re-reading the manifest cannot conjure a bucket.     */                                     \
  X(AwsErrorBucketNotFound, LOON_AWS_ERROR_BUCKET_NOT_FOUND, aws_bucket_not_found, LOON_ERROR_CATEGORY_CONFIG,         \
    "NoSuchBucket")                                                                                                    \
  /* A vortex file will not parse: flatbuffer/protobuf decode failure, a serde  */                                     \
  /* error, an offset outside the file, or a file too short to hold its EOF     */                                     \
  /* trailer. Mostly minted in our Rust bridge, which is the only layer holding */                                     \
  /* a typed VortexError and therefore the only one able to tell "the bytes are */                                     \
  /* wrong" from "the read failed" -- by the time C++ sees it, it is a string.  */                                     \
  /* The C++ reader mints it too, for the two shapes it can judge without      */                                      \
  /* parsing: an object shorter than the trailer, and a footer descriptor that  */                                     \
  /* points outside the file. A RECORDED size that contradicts an intact object */                                     \
  /* is NOT this -- that is ManifestCorrupted, because the bytes are fine and   */                                     \
  /* the metadata is not.                                                       */                                     \
  X(VortexFileCorrupted, LOON_VORTEX_FILE_CORRUPTED, vortex_file_corrupted, LOON_ERROR_CATEGORY_CORRUPTED, "")
