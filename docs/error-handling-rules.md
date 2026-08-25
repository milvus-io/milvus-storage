# milvus-storage Error Handling Rules v1

The single standard for error handling in this repository. Every review, new
producer and refactor is judged against it. Rule IDs are stable and are cited
directly in review findings (e.g. `violates R2.3`).

---

## 0. Model

A failure must answer four questions, and it must answer them through
**structured fields only** — never through message text:

| Question | Carried by | Values |
|---|---|---|
| Whose fault is it? | `ErrorCategory` | `User` / `System` (operator config and our own bugs) |
| Does retrying help? | `ErrorCategory` | `Retryable` (the observed cause is transient); everything else is not |
| Is business coordination needed? | `ErrorCategory` | `Conflict` — generic retry helpers must NOT replay |
| Is the data broken? | `ErrorCategory` | `DataFormat` |
| What exactly happened? | `ExtendStatusCode` / `LOON_*` | fine-grained diagnostic code |

`category` is a coarse fact/hint for generic callers; `code` is fine-grained
diagnosis for humans and specialised callers. `Unknown` exists only for
consumer-side forward compatibility — no producer may emit it.

---

## R1 — Distinguishability: retryable / non-retryable / user error

**R1.1 (classify at boundaries)** Every failure crossing a module boundary must
be distinguishable by `category`. A lower layer may not know the details, but it
must not hand the upper layer nothing but a string.

**R1.2 (conservative classification — producer owns classification)** Tag only
signals the producer **positively identifies**. Anything else stays untagged and
lands in the non-retryable System bucket. **Never invent retriability.**
> Violation: tagging every unknown HTTP 5xx as transient; treating
> `InvalidInput` as a caller error when the input was assembled internally.

**R1.3 (limits of `Retryable`)** `Retryable` means "the observed cause was
transient". It does **not** promise that replay is idempotent. Stateful objects
(writers, multipart uploads, transactions) must be destroyed and recreated after
a failure, and that must be enforced by a mechanism (e.g. `WriterStatus`), not
by a comment.

**R1.4 (Conflict is not Retryable)** Failures that need re-read / rebase /
submission of new work are `Conflict`; generic retry helpers must skip them and
`loon_ffi_is_retryable_errcode` must return false.

**R1.6 (a structural guard outranks a correct classification)** Where acting on
a misclassification could lose or corrupt data, the consumer must ALSO carry a
check that fails closed without depending on the classification being right.
Classification is a judgement about someone else's error; a guard is arithmetic
about our own state.
> The Lance writer treated "dataset is missing" as permission to create one.
> A wrong verdict there did not produce a bad error message: it produced a
> manifest entry pointing at another writer's fragment. The fix that matters is
> the fragment-count check after the write, not the sharper verdict before it.

**R1.5 (ownership is decided where provenance is known)** Whether a failure is
the user's or the operator's may only be decided at an entry point that knows
where the location and credentials came from. Lower layers always report
`System` plus a precise code; the entry point re-tags through the single
designated helper (`ExternalSourceErrorCodeFromStatus()`). Lower layers must not
guess ownership; entry points must not hand-roll the re-tag.

---

## R2 — No broad bucket swallowing narrow facts

**R2.1 (no new buckets)** A failure with a known cause must not degrade into a
bare `Status::Invalid` / `Status::IOError` / `Status::UnknownError`. New producer
code attaches an `ExtendStatusDetail` whenever the cause is known.

**R2.2 (one code, one meaning)** A code carries exactly one condition. Merging
several conditions into one code (e.g. S3 answering a missing key with 403 so
not-found and access-denied cannot be told apart) requires a written reason in
the X-macro table.

**R2.3 (classification may only get sharper)** Propagating, wrapping or
translating a status must never make it coarser. Already-classified statuses
keep their code and detail, and translating twice must be idempotent.
> Violation: `Invalid + DataCorrupted` → `IOError`; a decoded
> `NotImplemented` flattened to `IOError` on a second pass.

**R2.3a (an errno is the operating system speaking)** A status carrying an errno
detail describes reaching the bytes, not the bytes themselves. It must never be
relabelled as `DataFormat`: a full disk or an exhausted descriptor table is
recoverable, and calling it corruption tells the caller to rebuild a file that
is intact.

**R2.4 (where `catch` is allowed)** `catch (...)` / `catch (std::exception&)` may
appear **only** at: FFI/JNI entry points, thread and async task boundaries,
cross-language bridge conversion functions, and **abandonment paths** — the
release steps governed by R2.7a and R2.7b, whose contract is to log and drop
because the caller is already handling a failure. `AbandonQuietly` is the
sanctioned form; a hand-rolled swallow elsewhere is not. Everywhere else the
catch must map to an explicit code (a C++ exception escaping an FFI entry
point is `LOON_GOT_EXCEPTION`; an internal invariant violation is
`InternalInvariantViolated`) and must not
swallow or mis-tag. Exceptions are never control flow inside library code.

**R2.4a (an unknown exception is never the caller's fault)** A `catch (...)` may
absorb an unknown exception — including an allocation failure — into an
**internal/unexpected** verdict. It must never absorb one into a **user-fault**
verdict: not `Status::Invalid`, not `LOON_USER_INVALID_ARGUMENT`, not
`PackedInvalidArgs`, not `StorageConfigInvalid`, not `LOON_SOURCE_INVALID`.

This is the one OOM rule that survives unchanged. Otherwise allocation failure
has exactly one name, `LOON_MEMORY_ERROR` (2): a foreseeable large-data
allocation reports `Status::OutOfMemory`, which the FFI surface translates to
code 2 and segcore `MemAllocateFailed` (2034); a `std::bad_alloc` from ordinary
bookkeeping is left to propagate and end the process. Machinery built to keep
working after a small allocation failed is what produced the one outcome worse
than a crash — a publish slot consumed twice, leaving every waiter blocked on a
future no later completion could claim. A crash is restartable; that hang is not.

The failure mode this rule prevents is quieter than a hang and harder to
diagnose: `Manifest::serialize` reported any non-avro exception as `Invalid`,
i.e. as the caller's manifest being malformed, which sends an operator to fix
data that was fine.

**R2.4b (fail-stop requires unsettlable shared state)** `std::terminate()` inside
a `catch` is a process-level policy change, and the bar for it is exact: the
unwinding exception must be about to leave **shared state that can never be
settled** — a singleflight marker already visible to followers, a completion
slot a waiter already holds, a publish future whose second claim would block
every waiter forever. Only then does a crash (restartable) beat letting the
error escape (an unrestartable hang). A function with no such state — a pure
transform, a helper that only formats a string — must instead report
`Status::OutOfMemory` and let the taxonomy carry it (`LOON_MEMORY_ERROR` →
segcore `MemAllocateFailed`): `Manifest::serialize` and the Azure block-id
helper are the shape of the correct answer, and `catch (bad_alloc)
{ terminate(); }` at such a site is a defect, not a policy.

The sanctioned fail-stop sites today, each holding unsettlable shared state:

* `format_reader_cache.cpp` (singleflight marker visible to followers: 105,
  138, 178, 189)
* `filesystem/util_internal.h` (completion slot held by waiters: 179, 201, 224)
* `format/async_tasks.h` (publish-slot/PromiseState claims: 105, 129, 150)

Adding a site to this list is a reviewable event: the PR must name the shared
state that cannot be settled and cite this rule. Deleting a `terminate()` in
favour of a returned status needs no such justification.

**R2.5 (no silent fallback)** A failure must not quietly become a default value
(`0`, `{}`, an empty collection, a skipped item). The only exception is an API
whose contract is an **optional capability**, which must return `NotImplemented`
and must have a test pinning that behaviour.

**R2.6 (fan-out stops at the first failure)** Batch and fan-out operations return
the first lower-layer failure they observe unchanged. They must not continue
submitting work, probe the dependency, wait for unrelated work, or collect
additional failures to build a synthetic verdict. Work already accepted by an
executor may finish, but failure handling itself must not issue more I/O.
Waiting for every child is valid only on the all-success path. Once any child
fails, the public future/result completes immediately; lifetime tracking and
cleanup for already-accepted work continue independently.

*First observed is not deterministic, and that is the trade.* When several
children fail — a timeout and an access-denied in the same multipart upload —
which one is reported depends on completion order, so the same broken
deployment can surface either category across runs. The alternative is worse:
making it deterministic means waiting for every child to settle (banned above)
and then minting an aggregate verdict no layer actually established. Consumers
get one real lower-layer failure, never a synthetic parent; a caller that needs
the full picture reads the logs, not the code. The synthetic-parent codes were
retired outright for exactly this reason (see the gaps table in
error-codes.md).

**R2.7 (stateful failures stop locally)** A writer keeps its first lower-layer
failure unchanged. Later calls return it without waiting for other children,
issuing cleanup requests, probing the dependency, or building aggregate
diagnostics. An asynchronous Flush/Close completes as soon as an accepted child
reports that failure; waiting for every accepted child is only part of the
all-success path. Destruction may release local ownership, but failure handling
must not add storage I/O — **except** for the resource release carved out by
R2.7a, which the FFI `*_destroy` entry points do issue. Read this sentence
together with R2.7a before flagging a release as a violation. A dependency error
is reclassified only from a
documented, typed variant; broad or ambiguous variants stay unclassified and
non-retryable.

**R2.7a (releasing a resource is not diagnosis)** R2.7 bans I/O that tries to
find out *what went wrong*. It does not ban releasing something this writer
created and only this writer can name. An S3 multipart upload is the case that
forced the distinction: its parts are billed, they do not appear in
`ListObjectsV2`, and no GC that walks object keys will ever see them, so a
writer that drops the upload id silently strands them until a lifecycle rule
someone may not have configured expires them.

The release lives in an explicit `Abort()`, never in a destructor. The
high-level writer `Abort()` methods owned by this library are `void` and
`noexcept`: failure cleanup preserves the failure already being handled rather
than replacing it with a cleanup result. An interface that already requires a
status return keeps that contract. In particular, Arrow's S3 output-stream
`Abort()` returns a remote cleanup failure to a caller that invokes it directly
and retains the multipart upload id for a retry; when failed `Close()` invokes
the same method for cleanup, it ignores that result and returns the original
failure. Both forms obey these rules:

1. It must be safe on an already-failed writer — no `WriterStatus::Check()`.
2. Failure cleanup must never replace the primary failure. A `void noexcept`
   writer abort logs and drops its cleanup result; a status-returning stream
   abort may return it only to an explicit public caller. Every foreign call in
   a `noexcept` implementation must be wrapped in `AbandonQuietly`, or an
   escaping exception calls `std::terminate`.
3. A successful release must be idempotent. A failed remote release keeps the
   resource identity so an explicit caller can retry it.
4. **After a *successful* `Close()` it must be a no-op in every respect,
   including the recorded status.** Idempotence alone does not give you this: a
   naively idempotent `Abort()` flips a writer that finished cleanly to
   `Cancelled`. Every implementation therefore checks `closed_` *before*
   recording the cancellation. This is the invariant to check first in review.

Note that "closed" is not spelled the same way everywhere. `VortexFileWriter`
sets `closed_` on every failure path, so there "finished cleanly" is
`closed_ && status_.ok()`; using a bare `closed_` there would make
`Abort()` skip a release it still owes.

Every layer that owns a writer forwards it: `FormatWriter::Abort` →
`ColumnGroupWriter::Abort` → `api::Writer::abort`, plus `SegmentWriter::Abort`,
`LobColumnWriter::Abort` (which deletes the LOB files) and
`PackedRecordBatchWriter::Abort`. The FFI `*_destroy` entry points call it,
because destroy-without-close is the C caller explicitly abandoning the writer
and is the last frame that knows the writer existed. Streams with nothing to
release say so in a comment rather than silently doing less than the name
promises (Azure block blobs, whose uncommitted blocks the service drops on its
own). The vortex bridge is covered without a new verb: it already calls
`loon_filesystem_writer_destroy` on every terminal path, and that entry point
aborts before releasing. Lance drives its own Rust `object_store`, so nothing on
this side can reach its uncommitted fragments; that is accepted rather than
fixed, because the lance writer is test-only (`#ifdef BUILD_GTEST`, and
`LanceFormat::create_writer` returns `NotImplemented` in a production build).

**R2.7b (two verbs, and exactly one of them must happen)** A writer is finished
by `Close()` or by `Abort()`, and dropping it is neither.

`Close()` means "I am done with this writer", not "finish the file". A healthy
writer is finalized and its file published. A writer that has **already failed**
is abandoned instead: `Close()` releases what it holds in the store and returns
the first failure unchanged. Abandoning publishes nothing, so this does not
contradict R1.3 — it stops the parts from outliving the caller. `Close()` on a
failed writer therefore cannot succeed; any code or comment that says "even if
close succeeds" for that case is wrong.

This is what makes the contract usable from a language with no destructors. Go,
Java and C callers cannot express RAII, so before this a caller that only ever
called `Close()` leaked on every failure path — which is every binding.

`Abort()` is the unconditional form: give up and release. Library writer aborts
report nothing; a status-returning storage interface may report an explicit
call's cleanup failure as described in R2.7a.

One helper exists so implementations stop forgetting:

* `AbandonQuietly(step, fn)` — run one release step and swallow whatever it
  throws. Required at every foreign-code boundary inside a `noexcept` function.
  Its whole contract is to catch and map to nothing, which is the one sanctioned
  exception to R2.4.

An FFI entry point that creates remote state but returns **no handle** must
release it on every failure exit itself, because nothing the caller holds could
do it. `loon_filesystem_write_file` is the worked example.

**R2.8 (retry inside this library is bounded and uniform, or it does not exist)**
R1.2 bans inventing retriability; it does not ban retrying. Where repository-owned
credential providers retry internally, the budget is one shared constant and
what happens afterwards is a typed report, not a second opinion. AWS SDK
providers retain the SDK's own policy because Storage does not replace that
credential chain.

The rule exists because repository-owned IMDS and web-identity clients used to
have different budgets, so how long a credential outage took to surface depended
on which provider path ran.

Two things a bounded internal retry must get right:

* **Resend the same request, not a drained one.** A retried POST re-sends a body
  stream the previous attempt already read to the end, so it goes out empty and
  is refused on its contents rather than retried on its merits. Rewind between
  attempts. Curl's `CURLOPT_SEEKFUNCTION` does not cover this — that is for curl
  replaying a request it is already inside, and it is not called when
  `MakeRequest` is entered again.
* **Account for what the caller behind the lock pays.** A retry budget spent
  while holding an exclusive lock is charged to every other thread. Size it
  against the lock's contention, not against the request.

---

## R3 — Clean: not a pile of patches

**R3.1 (one transport per boundary)** Carrying classification across an FFI or
language boundary uses **the universal message marker**
(`__LOON_RUST_BRIDGE_ERRCODE__=<code>; message`): the Rust side embeds the code
at the error's construction, the code rides wherever the error text rides, and
the C++ side parses it back out. **No new string-marker schemes** and no
side-channel schemes: a thread-local slot that records beside the error was
retired because a verdict recorded on a tokio worker thread cannot be read from
the calling thread, an unclassified construction wiped a verdict recorded
earlier in the same call, and a stale verdict could attach to a later failure.
The accepted trade-off of the marker is that message text containing it
dictates the classification; the marker must never be user-suppliable data.
Paimon's former private markers (`[paimon:error=...]`) were retired onto the
shared marker. The former Paimon `data-invalid` marker was removed because the
dependency variant was too broad to prove data corruption.

**R3.2 (one decoder)** Each transport has exactly **one** decoder implementation
on the C++ side. New formats reuse it; copying it into another
`MakeXxxBridgeErrorStatus` is a violation.

**R3.3 (call sites do not infer)** No call site may hand-roll classification
(`if (status.IsInvalid()) fallback = A; else if (...) fallback = B;`). A fallback
expresses only "who owns an unclassified failure at this entry point"; missing
semantics are fixed in the producer. Error-message substrings are never evidence
for a code, HTTP status, retriability, or missing-object verdict.

**R3.4 (fix the producer)** Misclassification is fixed where the error is
produced. A consumer-side special case requires a comment explaining why the
producer cannot do it.

**R3.5 (table-driven)** Adding a code touches only the single X-macro table
(`LOON_*_CODE_LIST`). Adding classification branches to other switches or if
chains is a violation.

**R3.6 (no parallel error paths)** The same operation must not have two error
mechanisms (one throwing, one returning `Status`). Everything crossing a Rust
bridge returns `arrow::Result` / `arrow::Status`; custom exception types are not
thrown across it.

---

## R4 — Usable by upper layers, separately

**R4.1 (structured, all the way out)** Every language binding exposes `err_code`
and `category` as **structured fields** (exception attributes, struct members),
not only interpolated into a message string.

**R4.2 (binding-level structured verdict)** Each binding exposes `err_code`,
`category`, and `retryable`. Bindings may use one exception type; callers must
not need provider-specific subclasses to make the generic handling decision.

**R4.3 (messages are for humans)** Messages are diagnostics. **No consumer may
branch on a substring.** Anything a program must decide on needs a field.

**R4.4 (no code without a decision)** Before adding a code, answer: "what will an
upper layer do differently because of it?" If nothing, merge it into an existing
code and keep the distinction in the message.

**R4.5 (retired numbers are not reused)** Code values are never renumbered or
reused. Retired codes keep a placeholder comment stating why. An exported
tombstone can be removed only at a deliberate ABI break; the pre-release break
removed the old `aws_*` aliases, while the non-AWS 54 tombstone remains.

There are two retirement shapes, and they behave differently at runtime:

* **Out of the X-macro table** (14, 54, 106, 116) — no name, no
  category, no producer. `loon_ffi_error_category` answers `Unknown`, and the
  value lands in the consumer's conservative default branch. Only 54 still has
  a hand-exported tombstone symbol.
* **Still in the table, but with no producer** (`PackedUnexpected` 55) — keeps
  a name and a category, so `loon_ffi_error_category(55)` answers `System`,
  **not** `Unknown`. What it lacks is a producer, and `check_error_table.py`
  fails the build if one appears (`RESERVED_PRODUCERLESS`).

Do not apply the first bullet's reasoning to a code in the second group; the two
are distinguished by table membership, not by whether a symbol is exported.

---

## R5 — Maintainable

**R5.1 (single source of truth)** Name, value, category and segcore landing of
every code come only from the X-macro table and the
`default`-less `ToSegcoreErrorCode` switch. Documentation is script-verified
(`check_error_table.py`); no second hand-maintained table.

**R5.2 (no dead codes)** Every code has a real producer under `cpp/src` and a
test asserting its landing (category, retriability, segcore code). CI enforces
both. The exception is the reserved values listed in R4.5, which must have
**no** producer — the six out-of-table retirements cannot acquire one by
construction (no name, no category, nothing to construct), and
`PackedUnexpected` (55) is policed by `check_error_table.py`'s
`RESERVED_PRODUCERLESS`, which fails the build if a producer appears.

**R5.3 (keep the taxonomy gate focused)** CI runs `check_error_table.py` to
verify the taxonomy table, exported bindings, documentation rows, producers,
and reserved values. Repository-wide source-pattern ratchets are separate
governance work and are not part of this error taxonomy mechanism.

**R5.4 (write down why)** Every non-trivial classification decision carries a
comment explaining it — especially marking something retryable, marking it the
user's fault, merging conditions, and **deliberately leaving it unclassified**.

**R5.5 (classification lives in one function)** Each producer's classification
logic lives in a single function (e.g. `tryMakeClassifiedExtendArrowError`,
`classify_lance_error`), never spread across call sites.

**R5.6 (tests follow the rules)** A classification change ships with (1) an
end-to-end producer→category assertion and (2) an assertion that an unclassified
failure still lands in the conservative bucket.

---

## Appendix A — Review checklist

Walk it in order; each item gets ✅ pass / ⚠️ risk / ❌ violation.

1. Can every new or changed failure be distinguished by the caller? (R1.1)
2. Is any retriability invented? Anything uncertain tagged transient/user? (R1.2)
3. Is anything retryable stateful, and is recreation enforced? (R1.3)
4. Is a Conflict treated as Retryable? (R1.4)
5. Is ownership decided only where provenance is known? (R1.5)
6. Any new bare `Invalid`/`IOError` hiding an existing code? (R2.1)
7. Any classification downgrade in propagation? Is translation idempotent? (R2.3)
8. Are `catch` sites legal? Anything swallowed? (R2.4)
8a. Could any `catch (...)` report an unknown exception as the caller's fault? (R2.4a)
9. Any silent fallback (0 / empty / skipped)? (R2.5)
10. Do batch failures stop without aggregate diagnostics or additional I/O? (R2.6)
10a. Does failure handling add storage I/O beyond a resource release? (R2.7, R2.7a)
10b. **For any new or changed writer:** does `Abort()` preserve the primary
    failure, and for library-owned `void`/`noexcept` aborts is every foreign call
    wrapped in `AbandonQuietly`? Is abort a total no-op after a successful
    `Close()` — including the recorded status? Does a failed status-returning
    remote release retain enough identity for an explicit retry? Does `Close()`
    abandon an already-failed writer rather than leaving it to the caller? Does
    a handle-less FFI entry point release what it created on every exit? (R2.7a, R2.7b)
11. Any second transport or second decoder introduced? (R3.1, R3.2)
12. Do call sites hand-roll fallbacks? (R3.3)
13. Does a new code touch only the X-macro table? (R3.5)
14. Do bindings expose code + category structurally? (R4.1, R4.2)
15. Does anything branch on a substring? (R4.3)
16. Does a new code change an upper-layer decision? (R4.4)
17. Are docs, gates and tests updated? (R5.1, R5.2, R5.6)
18. Are the classification reasons written down? (R5.4)

## Appendix B — Severity

| Level | Definition | Example |
|---|---|---|
| **P0** | Data loss/corruption, or replaying an operation that is not safe to replay | Conflict tagged Retryable; a silent fallback turning a failure into an empty result |
| **P1** | Upper layer cannot make the right decision, or a compatibility regression | Classification downgrade (R2.3); valid data rejected as corrupt; first-write path misread as failure |
| **P2** | Inaccurate classification or leakage, upper layer can still cope | User error reported as system error; internal marker leaking into a user-visible message |
| **P3** | Consistency, maintainability, test quality | Two codes for one condition; tests using `ValueOrDie` and aborting |

## Appendix C — Known deviations (v1 baseline)

Recorded when the rules were written; reviews track whether these get worse.
Status as of the taxonomy pass that followed.

- **R3.1 / R3.2** — **Closed, then superseded by design.** There is one
  decoder table (`bridge_error.cpp`) and one transport: the universal message
  marker, embedded by the Rust producer at error construction and parsed by the
  shared decoder at the boundary. The typed thread-local side channel that
  v1 preferred was retired after it proved unable to carry a verdict across a
  thread hop (every tokio path, including credential resolution) and able to
  leak a stale verdict into a later failure; R3.1 records the trade-off.
- **R2.1** — bare `Status::Invalid` / `IOError` under `cpp/src`. **Mixed**:
  437/76 bare against 142 classified at baseline; 373/67 against 228 now. The
  classified count is up, but so is bare `Invalid` — new code is still being
  written unclassified faster than old code is being converted. The plan/read
  split in paimon and iceberg is closed; `segment` (0 bare, 26 classified) and
  `lob_column` (1/19) are done. What remains is concentrated in `format/` (230
  bare) and `filesystem/` (119), plus the `ffi/` argument checks that mint their
  own codes and never carry a detail.

  These five numbers are hand-maintained and nothing verifies them, which is the
  one place this document does what R5.1 forbids. Re-measure before citing them.
- **R4.1 / R4.2** — **Python is done**: `err_code`, `category`, and `retryable`
  are attributes on one `FFIError`; caller-owned input errors use
  `InvalidArgumentError`, with `tests/test_error_taxonomy.py` pinning the category values against the
  enum the native library exports. Java/JNI is **open** — it distinguishes only
  `IllegalArgumentException` (for `LOON_USER_INVALID_ARGUMENT` and
  `LOON_INVALID_PROPERTIES`) from `RuntimeException`, and branches on neither
  category nor code. Milvus's Go adapter is also open: it discards `err_code`
  and wraps every failure in one transient sentinel. The C ABI already exports
  everything both need.
## Appendix D — Decisions

Settled questions. Reviews apply these without re-litigating them.

**D1. One not-found convention per layer.** The errno channel
(`LOON_FILE_NOT_FOUND`, 12) belongs to the filesystem layer and to the vortex
fork that already emits it. Every bridge reports a missing object through the
taxonomy channel (`StorageNotFound`, 104). Both are still decoded, both land on
`ObjectNotExist`, and a test pins that equivalence — two numbers for one
condition is not a second meaning, it is two places for a consumer to remember.

**D2. Our own bugs do not look like storage failures.**
`InternalInvariantViolated` (122) is the attachable code for "this library's
invariant was violated": a closed reader reused, an internal index out of range,
an unreachable branch reached. It lands on segcore `UnexpectedError` (2001),
and `PackedInvalidArgs` were moved there with it. `PackedUnexpected` is a
legacy compatibility value; new packed internal failures use
`InternalInvariantViolated`. System
category, never User: the C ABI caller did not cause it, and no retry resolves
it.

**D3. Allocation failure names the node, not the store.** `LOON_MEMORY_ERROR`
(2) is the one FFI code for OOM, and it lands on segcore
`MemAllocateFailed` (2034) rather than `StorageError` — same answer as the
direct-C++ path.

It is `Retryable`, and it is the one code whose transient cause is a *condition*
rather than an observed service fault. That is consistent with R1.3, which asks
only whether the observed cause was transient: memory pressure passes, and the
request itself never reached a dependency, so replaying it — or rerouting to a
replica with headroom — is both safe and the only thing that can help. Contrast
R1.4: a Conflict replayed unchanged is *guaranteed* to fail again until the
caller re-reads and rebases, which is why it stays non-retryable.

This is also the only classification that agrees with the consumer. milvus's
`merr` marks 2034 `retriable: true` (`pkg/util/merr/segcore.go`); calling it
System here made `loon_ffi_is_retryable_errcode` and the segcore landing give
opposite answers for the same failure. Any change to this code's category has to
move `merr` with it, or move the segcore landing off 2034.

**D4. Codes may share a segcore landing when the producing subsystem is the
triage entry point.** Two groups do:

- DataFormat -> `DataFormatBroken`: `PackedMetadataCorrupted`,
  `PackedFileCorrupted`, `DataCorrupted`, `VortexDataFormat`.
- Our own bugs -> `UnexpectedError`: `PackedInvalidArgs`,
  `InternalInvariantViolated`; `PackedUnexpected` remains only for legacy
  compatibility.

Within a group the handling is identical, so R4.4 would say merge them; what
keeps them apart is that "which subsystem said this" is the first question
anyone debugging asks, and the code answers it without parsing a message. That
is the ONLY accepted reason to share a landing, it must be written in the
X-macro table, and it does not extend to codes whose difference is merely
descriptive.

**D5. Conflict has no segcore home.** `StorageConflict`,
`StoragePreConditionFailed`, `TxnExhaustedRetry` and `TxnResolutionFailed` land
on `StorageError` because segcore has no coordination code. Known expressiveness
gap, tracked outside this repository; do not paper over it by making them
retryable.

**D6. When nothing below us identifies corruption, nobody invents it.** Vortex
reports footer-deserialization failures through its catch-all `Other` variant
(`vortex_bail!` in vortex-file/src/footer/deserializer.rs; the `VortexError`
enum has no corrupt-data variant at all), so no variant match can see them.
Three answers were considered: infer it in the format reader from its position
in the retry loop (what used to happen, and what R3.4 forbids), infer it in the
bridge at the `open()` call site (defensible -- that call only reads the tail
and deserializes the footer -- but it needs a list of variants to exclude, and
the list is a standing liability every time vortex adds one), or let it stay
unclassified. We chose unclassified.

The cost is real and accepted: a corrupt vortex file reaches segcore as
`StorageError` rather than `DataFormatBroken`, so a caller cannot tell it apart
from a storage failure and may retry a file that will never read. What is bought
is that no layer claims a verdict nobody established, which is R1.2 applied to
ourselves rather than only to producers below us. `VortexDataFormat` is still
produced everywhere vortex DOES give a typed error, and getting a type for this
one upstream (#1732, #7483) is the fix that would let us classify it honestly.
