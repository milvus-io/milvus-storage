# milvus-storage error codes

Every error milvus-storage returns carries a **code**, and each code is one
statement of fact about what happened. A **category** is the same statement,
coarser, for a caller that does not want to switch over every code.

| category | what it means |
|---|---|
| **User** | the location string we were handed does not name a usable object |
| **Config** | the deployment's own settings or credentials are unusable |
| **Transient** | a condition that may clear on its own -- timeout, throttle, 5xx |
| **Conflict** | someone else changed the object between our read and our write |
| **Missing** | the named object is not there |
| **Corrupted** | a layer parsed the bytes and found them wrong |
| **Permanent** | a bug on our side, or data that is gone |

## What this does NOT tell you

There is no retry predicate here, and that is deliberate.

Whether an operation may be redone depends on the operation, and this library is
never told what the caller was doing. The same `NoSuchUpload` means "start a new
upload" to the layer that owns the write and nothing at all to a layer that was
only reading. The same conditional-write conflict means "re-read and rebase" for
a manifest commit and "just send it again" for an idempotent delete. A boolean
returned from here would have to guess which of those the caller meant, and a
wrong guess is either a retry loop that can never win or an abandoned operation
that would have succeeded.

The same applies to who should hear about a failure. `Config` says the
deployment's settings are unusable, not that an operator must be paged; whether
that is an alert, a log line, or an error returned to a user is a decision that
depends on the caller's context.

So: match on the code when you need precision, on the category when you do not,
and decide using what you know about your own request.

`Unknown` is the eighth value and a real answer. A producer says it when this
library could not determine what happened -- an unclassified failure arriving
from below. A consumer says it for a code newer than itself. Neither is evidence
that the condition is permanent, which is precisely why it is not called
permanent: treating the two as synonyms is what this taxonomy was written to
stop.

**A batch answers with one category or nothing.** If every member failed the
same kind of way, that is a fact about the batch and it is reported. If they
disagree, or if any member arrived with no classification, the batch is
`Unknown`. Either way every individual failure is named in the message.

Two other rules were tried first. Reporting whichever member failed *first* let
a throttle that finished early hide a denial that finished late, so half the
time a caller was told to retry an operation containing something no retry could
fix. Ranking the categories and reporting the worst fixed that, but only holds
if every call site feeds every failure into the ranking -- and one did not, so a
mix of unclassified blobs and a 503 answered "transient", breaking the rule in
the very place it was meant to apply. The rule above cannot be implemented
halfway: an unclassified member is a disagreement by construction.

**`Missing` and `Corrupted` each carry a discipline, and both are machine-checked.**
Only a producer holding a *definitive* not-found from the store may say Missing;
only a producer that actually *parsed* the bytes and found them wrong may say
Corrupted. Neither may be inferred from an unclassified failure -- which is why
the coarse arrow-status fallback lands on a generic storage failure rather than
guessing corruption. `CoarseFallbackNeverClaimsCorruption` pins that: of the
~380 unclassified `Status::Invalid` sites in `cpp/src`, almost none are corrupt
bytes, and an alert that is mostly false is worse than no alert.

**`User` is deliberately narrow.** Only an entry point contractually handed a
user-supplied location can know the input came from a user, so nothing below
those entry points may classify itself `User`; the taxonomy test fails the build
if any producer-side code tries.

**There is no `Throttled` category.** Rate limiting keeps its own *code*
(`StorageTransientThrottling`, 109) so it can be logged and measured on its own,
but it is not a separate category, because the thing that would distinguish it
-- how long the store told us to wait -- is never extracted anywhere in this
repository and `LoonFFIResult` has no field that could carry it.

The table below is generated from `LOON_INTERNAL_ERROR_CODE_LIST` and
`LOON_EXTEND_STATUS_CODE_LIST` in
[`cpp/include/milvus-storage/ffi_internal/ffi_error_code.h`](../cpp/include/milvus-storage/ffi_internal/ffi_error_code.h),
which is the single source for the exported `loon_errcode_*` constants,
`error_to_string`, `loon_ffi_error_category`, `enum class ExtendStatusCode` and
its metadata. Adding a code there is the only edit needed; failing to classify
it fails `cpp/test/ffi/error_taxonomy_test.cpp`.

## How to consume it

```c
// Success first: loon_ffi_error_category(LOON_SUCCESS) is UNKNOWN, because the
// category function answers "which kind of failure", and success is not one.
if (loon_ffi_is_success(&result)) {
  /* ... use the result ... */
  loon_ffi_free_result(&result);
  return;
}

// if/else, not switch: the loon_error_category_* symbols are `extern const int`
// (see ffi_c.h), and C requires case labels to be integer constant expressions.
int category = loon_ffi_error_category(result.err_code);
// ... branch on category, or on result.err_code when you need the exact
// condition. What to DO about it is yours to decide: you know the operation,
// this library does not.

/* Always, on every path. The message is strdup'd by the library, so a loop that
   classifies and continues without freeing leaks once per attempt. */
loon_ffi_free_result(&result);
```

`loon_ffi_error_category` is a pure function of the code, so it needs no room in
`LoonFFIResult` and a consumer can classify a code it has never seen.

## The codes

### FFI-layer codes (1–14)

Minted by the FFI entry points themselves (argument checks, catch-all handlers). They never
carry an `ExtendStatusDetail`.

| Code | Value | Category | S3 / OSS equivalent | Raised when |
|---|---|---|---|---|
| `LOON_INVALID_ARGS` | 1 | Permanent | `InvalidArgument` | Null pointer, empty path, bad range across the C ABI |
| `LOON_MEMORY_ERROR` | 2 | Permanent | — | Local allocation failed. Kept distinct from `LOON_GOT_EXCEPTION` so an OOM stays diagnosable as an OOM rather than being filed under "some exception" |
| `LOON_ARROW_ERROR` | 3 | Unknown | `InternalError` | Unclassified arrow failure — the conservative fallback |
| `LOON_LOGICAL_ERROR` | 4 | Permanent | `InternalError` | Internal invariant violated (our bug) |
| `LOON_GOT_EXCEPTION` | 5 | Permanent | `InternalError` | A C++ exception escaped an FFI entry point |
| `LOON_UNREACHABLE_ERROR` | 6 | Permanent | `InternalError` | Reached code that should be unreachable |
| `LOON_INVALID_PROPERTIES` | 7 | Config | `InvalidArgument` | Registered deployment/storage properties are invalid, or the FFI property map is malformed |
| `LOON_FAULT_INJECT_ERROR` | 8 | Permanent | — | Test-only fault injection |
| `LOON_NOT_SUPPORT` | 9 | Config | `NotImplemented` | The caller asked for a feature/format this build lacks |
| `LOON_FILE_NOT_FOUND` | 12 | Missing | `NoSuchKey` | An object on an **internally generated** path is missing (GC race, lost data, stale metadata) |
| `LOON_SOURCE_INVALID` | 13 | **User** | `NoSuchBucket` / `AccessDenied` | The external source the user named is unusable: it does not exist, or the credentials they supplied were rejected. One code, not two -- S3 answers a missing key with 403 rather than 404 when the caller lacks `s3:ListBucket`, so the split cannot be made accurately here |

### Codes that travel on an `arrow::Status` (50–112)

These are `ExtendStatusCode` values: a producing layer attaches one as an `ExtendStatusDetail`
and it survives to the FFI boundary.

| Code | Value | Category | S3 / OSS equivalent | Raised when |
|---|---|---|---|---|
| `PackedInvalidArgs` | 50 | Permanent | `InvalidArgument` | Bad arguments to a packed API |
| `PackedStorageIO` | 51 | Unknown | — | Packed-layer IO failure |
| `PackedMetadataCorrupted` | 52 | Corrupted | — | Packed metadata does not parse |
| `PackedFileCorrupted` | 53 | Corrupted | — | Packed file body is corrupt |
| `PackedArrowError` | 54 | Permanent | — | Arrow failure inside packed |
| `PackedUnexpected` | 55 | Permanent | `InternalError` | Packed internal error |
| `AwsErrorNoSuchUpload` | 101 | Missing | `NoSuchUpload` | Multipart upload state is gone |
| `AwsErrorConflict` | 102 | **Conflict** | `OperationAborted` | A conflicting operation is already in progress on the object |
| `AwsErrorPreConditionFailed` | 103 | **Conflict** | `PreconditionFailed` | A conditional write's precondition no longer held |
| `AwsErrorNotFound` | 104 | Missing | `NoSuchKey` | An object on an internally generated path is gone. A missing BUCKET is no longer this -- it is `AwsErrorBucketNotFound` (118), Config, because re-reading metadata cannot produce a bucket |
| `AwsErrorAccessDenied` | 105 | Config | `AccessDenied` / `InvalidAccessKeyId` / `SignatureDoesNotMatch` | Credentials or permissions wrong |
| `StorageTransientNetwork` | 107 | **Transient** | — | Connection reset / refused / aborted |
| `StorageTransientTimeout` | 108 | **Transient** | `RequestTimeout` | Request timed out |
| `StorageTransientThrottling` | 109 | **Transient** | `SlowDown` / `TooManyRequests` (429) | Object store throttled us |
| `StorageTransientService` | 110 | **Transient** | `ServiceUnavailable` / `InternalError` (5xx) | Object store returned a server error |
| `StorageTransientUnspecified` | 120 | **Transient** | — | The store failed in a way it is expected to recover from, and no cause could be named. The SDK's own retry verdict is the only signal; it says the condition clears without saying why |
| `TxnExhaustedRetry` | 111 | **Conflict** | — | Manifest transaction spent its own retry budget |
| `TxnResolutionFailed` | 112 | **Conflict** | — | Manifest merge/resolution failed |
| `StorageConfigInvalid` | 115 | Config | `InvalidArgument` | Deployment storage config is unusable: unknown cloud provider or storage type, malformed `extfs.*` property |
| `ManifestCorrupted` | 117 | Corrupted | — | The manifest does not parse: bad MILV magic, truncated stream, avro body that does not decode. Also used when a size the manifest RECORDED contradicts an intact object -- the bytes are fine, so blaming them would have a good file quarantined |
| `AwsErrorBucketNotFound` | 118 | Config | `NoSuchBucket` | The bucket the deployment names does not exist — not data loss, and no amount of re-reading metadata produces one |
| `VortexFileCorrupted` | 119 | Corrupted | — | A vortex file does not decode: flatbuffer/protobuf failure, serde error, an offset outside the file, or a file too short to hold its EOF trailer. Mostly classified in our Rust bridge — the only layer holding a typed `VortexError`, since by the time C++ sees the failure it is a string. The C++ reader mints it for the two shapes it can judge without parsing: a file shorter than the trailer, and a footer descriptor pointing outside the file |

## Alignment with AWS S3 / Aliyun OSS, and where it deliberately differs

The vocabulary follows the S3 REST error codes (Aliyun OSS, Tencent COS, Huawei OBS and MinIO
use the same names for these conditions), and the default rule is S3's own split: **4xx is the
caller's problem, 5xx and 408/429 are ours and retriable**. Eight places break that rule on
purpose — each is pinned by a test in `error_taxonomy_test.cpp`.

| # | Code | AWS says | We say | Why |
|---|---|---|---|
| 1 | `AwsErrorNoSuchUpload` (101) | 404, client error | **Missing** | The upload id the caller held is gone. What helps is a NEW upload, which only the layer that owns the write can start; we report that the handle is dead and leave that decision there. |
| 2 | `AwsErrorNotFound` (104) | 404, client error | **Missing** | On an internally generated path the caller never chose the key, so it is not their error; and it is not Permanent either, because re-reading the manifest may show the file was legitimately collected. The user-supplied counterpart is `LOON_SOURCE_INVALID` (13), which *is* a user error. |
| 3 | `AwsErrorAccessDenied` (105) | 403, client error | **Config** | The credentials are operator configuration, not part of the caller's request — so it is neither the caller's fault (User) nor something to file as a generic storage failure (Permanent). The user-supplied counterpart is `LOON_SOURCE_INVALID` (13). |
| 4 | `LOON_LOGICAL_ERROR` (4), `LOON_GOT_EXCEPTION` (5), `LOON_UNREACHABLE_ERROR` (6) | `InternalError` (500) is retriable | **Permanent** | These are our own bugs, not a condition that clears. `LOON_ARROW_ERROR` (3) is deliberately NOT in this list: it is the code for a failure nobody classified, and calling that permanent is the invention this taxonomy exists to avoid. |
| 5 | `StorageTransientTimeout` (108) | `RequestTimeout` is 400 (4xx) | **Transient** | AWS itself retries `RequestTimeout`; the 4xx status is a historical quirk, not a statement about ownership. |
| 6 | `AwsErrorConflict` (102), `AwsErrorPreConditionFailed` (103), `TxnExhaustedRetry` (111), `TxnResolutionFailed` (112) | 409/412 are client errors; the transaction codes have no S3 name | **Conflict** | Lost a race, not made a bad request — so not `User`; and the object's state changed under us, which `Transient` does not say. What to do about it depends on the operation, and the caller knows that. |
| 7 | `AwsErrorBucketNotFound` (118) | grouped with `NoSuchKey` as a 404 | **Config, split out** | Nothing was lost and no re-read produces a bucket. The deployment points at something that is not there — a configuration fix, landing on `2016 BucketInvalid`, a milvus code we already had and were not using. |
| 8 | the coarse fallback | arrow's `Invalid` suggests bad data | **`StorageError`, not `DataFormatBroken`** | An unclassified `Status::Invalid` is overwhelmingly a null-pointer precondition, missing config or a caller contract violation — not corrupt bytes. `Corrupted` has exactly one source: a producer that actually parsed the bytes. |

Conditions with **no object-storage counterpart** (the `—` rows above): the packed-layer codes
(51–55), corruption (52, 53), the manifest-transaction codes (111, 112), local memory (2) and
fault injection (8). These live below or beside the object-store API, so S3 has no name for
them. An empty `s3_code` in the table is the documented way of saying "no counterpart".

### The one classification that depends on the call site

`not-found` and `access-denied` are the same object-store condition with two different owners:

- an object on a path **milvus generated** → not the caller's problem (`AwsErrorNotFound` 104,
  a **Missing** failure -- re-read the metadata and decide, since this layer cannot tell a GC
  race from real loss / `AwsErrorAccessDenied` 105, a Config failure);
- a bucket or key the **user typed** into an external-source definition → user error
  (`LOON_SOURCE_INVALID` 13, which covers both -- see the code's own comment for why
  not-found and access-denied cannot be split accurately at this layer).

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
| Vortex bridge | **Partly** — I/O errors round-trip their FFI code through the C++ shim, and decode failures that name a contradiction inside the file (bad magic, truncated or trailing metadata, a child count the layout does not match) classify as `VortexFileCorrupted`. Other decode bails stay unclassified on purpose: the footer reader deliberately provokes some of them on healthy files, so tagging them would mint quarantine orders for intact data |
| Lance / Iceberg bridges | In flight — see [#597](https://github.com/milvus-io/milvus-storage/pull/597) |
| Packed layer | Partial: `Packed*` (50–55) |
| Azure filesystem | **Yes** — `ClassifyAzureError` maps HTTP status and error code onto the taxonomy; unrecognized responses stay unclassified rather than being guessed at |
| Local filesystem | **Almost none** — one classified case |
| Paimon (Rust `FileIO` / opendal, a separate IO stack that never reaches the C++ classifier) | **No** |
| Format layer (`Status::Invalid` for schema/type problems) | **No** — sampling them showed most are internal contract violations, not corrupt data, which is why nothing infers corruption from them. They are still unclassified: the caller learns only that storage failed. |
| Filesystem config and URI parsing (`fs.cpp`) | **Yes** — `StorageConfigInvalid` (115), covering both the property map and the URI |

Anything not classified arrives as `LOON_ARROW_ERROR` (3) / plain `arrow::Status`, which
says only that storage failed. That is conservative by design, but it means the absence of a
transient code is not evidence that a failure is permanent — the rows marked **No** are gaps
in coverage, not verdicts.
