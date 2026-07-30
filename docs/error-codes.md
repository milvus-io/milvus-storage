# milvus-storage error codes

Every error milvus-storage returns to an upper layer carries a **category**, and
the category is chosen so that each one maps to exactly **one consumer action**.
That is the test for whether a category earns its place: if two conditions call
for the same action they belong together; if one needs a different action it
needs its own category.

| category | consumer action | retry |
|---|---|---|
| **User** | return it to the caller; do not alert | no |
| **Config** | alert an operator; do not blame the caller | no |
| **Transient** | retry with normal backoff | yes |
| **Throttled** | back off per Retry-After **and shed concurrency** — a normal retry makes it worse | yes |
| **Conflict** | **re-read state, rebase, re-submit** — replaying the same bytes fails identically | yes |
| **Permanent** | alert a developer | no |

Two pairs look redundant and are not:

- **User vs Config** both mean "do not retry", but differ in *who fixes it*.
  Reporting a misconfigured endpoint as the caller's fault sends the user
  editing their query forever, and never pages the person who can fix it.
- **Transient vs Throttled vs Conflict** are all retriable, but a single retry
  strategy is wrong for two of them: retrying into a throttling store amplifies
  the overload, and replaying a conditional write that lost a race fails
  identically every time.

Retriability is therefore **derived, never stored**:

```
retryable == (Transient || Throttled || Conflict)
```

`Unknown` is not a seventh kind of error and no producer emits it. It is what a
**consumer** reports for a code newer than itself, and it must be handled as
`Permanent`: never retry what you cannot classify.

Consumers branch on the **category** (closed, six values, exhaustive). The
**code** is for diagnosis — logs, metrics, and the finer policy decisions a
category cannot express (which backoff curve, which alert route). New codes can
be added without breaking a consumer, because every code belongs to one of the
six.

The table below is generated from `LOON_INTERNAL_ERROR_CODE_LIST` and
`LOON_EXTEND_STATUS_CODE_LIST` in
[`cpp/include/milvus-storage/ffi_internal/ffi_error_code.h`](../cpp/include/milvus-storage/ffi_internal/ffi_error_code.h),
which is the single source for the exported `loon_errcode_*` constants, `error_to_string`,
`loon_ffi_error_category`, `loon_ffi_is_retryable_errcode`, `enum class ExtendStatusCode` and
its metadata. Adding a code there is the only edit needed; failing to classify it breaks the
build (`ToSegcoreErrorCode` is a no-`default` switch under `-Werror=switch`) or
`cpp/test/ffi/error_taxonomy_test.cpp`.

## How to consume it

```c
int category = loon_ffi_error_category(result.err_code);
if (category == loon_error_category_transient) {          // retry
} else if (category == loon_error_category_user) {        // report to the requester, never retry
} else {                                                  // permanent or unknown: fail and alert
}
```

`loon_ffi_error_category` is a pure function of the code, so it needs no room in
`LoonFFIResult` and a consumer can classify a code it has never seen. An unrecognized code
returns `loon_error_category_unknown`, which **must** be treated as permanent — never retry a
failure you cannot classify.

C++ consumers linking directly get the same verdict through `ToSegcoreError(status)` →
`milvus::ErrorCode`, where milvus's `merr` treats `2045` as retriable and `2042` as a
caller-input error.

## The codes

### FFI-layer codes (1–14)

Minted by the FFI entry points themselves (argument checks, catch-all handlers). They never
carry an `ExtendStatusDetail`.

| Code | Value | Category | Retry | S3 / OSS equivalent | Raised when |
|---|---|---|---|---|---|
| `LOON_INVALID_ARGS` | 1 | **User** | no | `InvalidArgument` | Null pointer, empty path, bad range across the C ABI |
| `LOON_MEMORY_ERROR` | 2 | **Transient** | **yes** | — | Local allocation failed; another node or a later attempt may have the memory |
| `LOON_ARROW_ERROR` | 3 | Permanent | no | `InternalError` | Unclassified arrow failure — the conservative fallback |
| `LOON_LOGICAL_ERROR` | 4 | Permanent | no | `InternalError` | Internal invariant violated (our bug) |
| `LOON_GOT_EXCEPTION` | 5 | Permanent | no | `InternalError` | A C++ exception escaped an FFI entry point |
| `LOON_UNREACHABLE_ERROR` | 6 | Permanent | no | `InternalError` | Reached code that should be unreachable |
| `LOON_INVALID_PROPERTIES` | 7 | **User** | no | `InvalidArgument` | The caller's properties are malformed or unusable |
| `LOON_FAULT_INJECT_ERROR` | 8 | Permanent | no | — | Test-only fault injection |
| `LOON_NOT_SUPPORT` | 9 | **User** | no | `NotImplemented` | The caller asked for a feature/format this build lacks |
| `LOON_FILE_NOT_FOUND` | 12 | Permanent | no | `NoSuchKey` | An object on an **internally generated** path is missing (GC race, lost data, stale metadata) |
| `LOON_SOURCE_NOT_FOUND` | 13 | **User** | no | `NoSuchBucket` | A path/bucket the **user named** does not exist |
| `LOON_SOURCE_ACCESS_DENIED` | 14 | **User** | no | `AccessDenied` | Credentials the **user supplied** for an external source were rejected |

### Codes that travel on an `arrow::Status` (50–112)

These are `ExtendStatusCode` values: a producing layer attaches one as an `ExtendStatusDetail`
and it survives to the FFI boundary *and* to segcore.

| Code | Value | Category | Retry | S3 / OSS equivalent | segcore | Raised when |
|---|---|---|---|---|---|---|
| `PackedInvalidArgs` | 50 | **User** | no | `InvalidArgument` | 2042 `InvalidParameter` | Bad arguments to a packed API |
| `PackedStorageIO` | 51 | Permanent | no | — | 2044 `StorageError` | Packed-layer IO failure |
| `PackedMetadataCorrupted` | 52 | Permanent | no | — | 2024 `DataFormatBroken` | Packed metadata does not parse |
| `PackedFileCorrupted` | 53 | Permanent | no | — | 2024 `DataFormatBroken` | Packed file body is corrupt |
| `PackedArrowError` | 54 | Permanent | no | — | 2044 `StorageError` | Arrow failure inside packed |
| `PackedUnexpected` | 55 | Permanent | no | `InternalError` | 2044 `StorageError` | Packed internal error |
| `AwsErrorNoSuchUpload` | 101 | **Transient** | **yes** | `NoSuchUpload` | 2045 `StorageTransientError` | Multipart upload state is gone |
| `AwsErrorConflict` | 102 | Permanent | no | `OperationAborted` | 2044 `StorageError` | Concurrent-modification conflict |
| `AwsErrorPreConditionFailed` | 103 | Permanent | no | `PreconditionFailed` | 2044 `StorageError` | Conditional write precondition failed |
| `AwsErrorNotFound` | 104 | Permanent | no | `NoSuchKey` / `NoSuchBucket` | 2017 `ObjectNotExist` | Object/bucket gone on an internal path |
| `AwsErrorAccessDenied` | 105 | Permanent | no | `AccessDenied` / `InvalidAccessKeyId` / `SignatureDoesNotMatch` | 2044 `StorageError` | Credentials or permissions wrong |
| `AwsErrorNonRetryable` | 106 | Permanent | no | — | 2044 `StorageError` | AWS SDK judged it non-retryable (`ShouldRetry() == false`) |
| `StorageTransientNetwork` | 107 | **Transient** | **yes** | `RequestTimeout` | 2045 `StorageTransientError` | Connection reset / refused / aborted |
| `StorageTransientTimeout` | 108 | **Transient** | **yes** | `RequestTimeout` | 2045 `StorageTransientError` | Request timed out |
| `StorageTransientThrottling` | 109 | **Transient** | **yes** | `SlowDown` / `TooManyRequests` (429) | 2045 `StorageTransientError` | Object store throttled us |
| `StorageTransientService` | 110 | **Transient** | **yes** | `ServiceUnavailable` / `InternalError` (5xx) | 2045 `StorageTransientError` | Object store returned a server error |
| `TxnExhaustedRetry` | 111 | Permanent | no | — | 2044 `StorageError` | Manifest transaction spent its own retry budget |
| `TxnResolutionFailed` | 112 | Permanent | no | — | 2044 `StorageError` | Manifest merge/resolution failed |

## Alignment with AWS S3 / Aliyun OSS, and where it deliberately differs

The vocabulary follows the S3 REST error codes (Aliyun OSS, Tencent COS, Huawei OBS and MinIO
use the same names for these conditions), and the default rule is S3's own split: **4xx is the
caller's problem, 5xx and 408/429 are ours and retriable**. Five places break that rule on
purpose — each is pinned by a test in `error_taxonomy_test.cpp`.

| # | Code | AWS says | We say | Why |
|---|---|---|---|---|
| 1 | `AwsErrorNoSuchUpload` (101) | 404, client error | **Transient, retriable** | Our retry is at the *operation* level: it starts a fresh multipart upload, which then succeeds. Retrying is genuinely useful here. |
| 2 | `AwsErrorNotFound` (104) | 404, client error | **Permanent system error** | On an internally generated path the caller never chose the key; a missing object means a GC race or lost data, which an operator must see. The user-supplied counterpart is `LOON_SOURCE_NOT_FOUND` (13), which *is* a user error. |
| 3 | `AwsErrorAccessDenied` (105) | 403, client error | **Permanent system error** | The credentials are operator configuration, not part of the caller's request. The user-supplied counterpart is `LOON_SOURCE_ACCESS_DENIED` (14). |
| 4 | `LOON_LOGICAL_ERROR` (4), `LOON_ARROW_ERROR` (3), `LOON_GOT_EXCEPTION` (5), `LOON_UNREACHABLE_ERROR` (6) | `InternalError` (500) is retriable | **Permanent** | These are our own bugs. Retrying reproduces them and turns a bug into a retry storm. |
| 5 | `StorageTransientTimeout` (108) | `RequestTimeout` is 400 (4xx) | **Transient, retriable** | AWS itself retries `RequestTimeout`; the 4xx status is a historical quirk, not a statement about ownership. |

Conditions with **no object-storage counterpart** (the `—` rows above): the packed-layer codes
(51–55), corruption (52, 53), the manifest-transaction codes (111, 112), local memory (2) and
fault injection (8). These live below or beside the object-store API, so S3 has no name for
them. An empty `s3_code` in the table is the documented way of saying "no counterpart".

### The one classification that depends on the call site

`not-found` and `access-denied` are the same object-store condition with two different owners:

- an object on a path **milvus generated** → system failure (`AwsErrorNotFound` 104 / `AwsErrorAccessDenied` 105);
- a bucket or key the **user typed** into an external-source definition → user error
  (`LOON_SOURCE_NOT_FOUND` 13 / `LOON_SOURCE_ACCESS_DENIED` 14).

The layer that raises the error cannot tell them apart — only the entry point knows where the
path came from. So the re-tagging lives at exactly the entry points that accept a user-supplied
location (`loon_exttable_explore`, `loon_exttable_get_file_info`), through
`UserSourceErrorCodeFromStatus()`. Do not call it anywhere else.

## Coverage: which layers actually produce classified codes

A taxonomy is only as good as the layers that populate it. Current state:

| Producer | Classified? |
|---|---|
| S3 filesystem — AWS, Aliyun OSS, Tencent COS, Huawei OBS, MinIO (`s3_internal.h`) | **Yes**, full: 101–110 |
| GCS (`GcpFileSystemProducer`, which builds an `S3FileSystem`) | **Yes**, inherits the S3 classifier |
| Vortex bridge | **Yes** — round-trips the FFI code through the C++ filesystem shim |
| Lance / Iceberg bridges | In flight — see [#597](https://github.com/milvus-io/milvus-storage/pull/597) |
| Packed layer | Partial: `Packed*` (50–55) |
| Azure filesystem | **No** — one classified case, the rest are plain `IOError`. See [#595](https://github.com/milvus-io/milvus-storage/issues/595) |
| Local filesystem | **Almost none** — one classified case |
| Paimon (Rust `FileIO` / opendal, a separate IO stack that never reaches the C++ classifier) | **No** |
| Format layer (`Status::Invalid` for schema/type/URI problems) | **No** — these reach segcore as `2024 DataFormatBroken`, i.e. user mistakes reported as data corruption |

Anything not classified arrives as `LOON_ARROW_ERROR` (3) / plain `arrow::Status`, and is
therefore treated as **permanent, non-retriable** — conservative by design, but it means the
absence of a transient code is not evidence that a failure is permanent. Closing the rows
marked **No** is what makes the retry verdict trustworthy end-to-end.
