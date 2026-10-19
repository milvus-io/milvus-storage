# Manifest index path roundtrip fix

## Failure and scope

On `main` revision `5fab6a311dca39414d977303c271c86f1250d693`, reading an
index expands its segment-relative directory into an object key. A subsequent
transaction retaining that index writes the full key unchanged when it is
outside `<segment>/_index/`. Reading that manifest prepends the segment again.
Publishing another index, updating stats or deltalogs, or deleting another
index can all trigger this without concurrency.

New indexes supplied by Milvus already use paths relative to
`<segment>/_index/`, including `..` components for `index_files` and `index_v1`.
Those inputs must remain distinguishable from indexes expanded during a read.

## Representation and compatibility

Retain the original stored path, expanded path, and source segment as internal
provenance on each index read by `Manifest::ReadFrom`. Copies of indexes and
manifests retain that provenance, including the copy in `applyUpdates`.
Writing an unchanged index to the same segment reuses its stored path exactly.
Changing its path invalidates the provenance by value comparison, so a new
relative path is not interpreted as a previously expanded object key.

When a loaded index is explicitly written into another segment's manifest,
an expanded object key is made relative to the new index base. This keeps its
existing target, without guessing the root directory or its depth. Copying an
Avro file to another segment continues to rebase its stored relative paths on
read. External URIs and local absolute paths keep the existing behavior.
URI coverage uses the existing `StorageUri` endpoint/bucket/key syntax (for
example `s3://endpoint/bucket/index`); this does not add support for URI forms
rejected by the current parser.

The Avro schema/version, C FFI layout, index identities, conflict resolution,
file locations, and full-path read results are unchanged. The C++ `Index`
layout gains internal provenance; C++ consumers must rebuild against matching
headers and library. Newly constructed indexes still follow the existing input
contract: a segment-relative path, an absolute local path, a URI, or the
previously supported full path under the current index directory. Arbitrary
unmarked object keys outside that directory are not a new input format.

The shared `ToRelative`/`ToAbsolute` helpers and column group, stats, deltalog,
and LOB conversions are unchanged. Both asynchronous transaction methods call
the synchronous implementation; no separate async path conversion is needed.
FFI additions construct fresh indexes, while retained indexes stay in the C++
transaction. Raw `serialize`/`deserialize` APIs remain raw codecs and do not
carry runtime provenance across a serialization boundary.

The existing reader normalizes relative paths lexically. Object keys containing
literal repeated separators, `.` or `..` therefore still have that limitation.
This fix preserves the stored representation on unchanged roundtrips; it does
not claim arbitrary object-key fidelity or introduce broader normalization.

## Historical manifests

This prevents new corruption; it does not repair already incorrect manifests.
An unchanged previously corrupted entry is preserved as stored, with the same
incorrect read target. No repeated prefix is stripped heuristically.
Repair requires the original manifests/version chain, layout and index IDs,
verified object existence, and a separately designed conditional rewrite with
cache/recovery checks and rollback. No UAT data or index files are modified.

## Regression coverage

- `TransactionTest.IndexPathsSurviveIndependentCommits`: real synchronous and
  asynchronous open/commit calls, empty/single/multiple-component roots, ten
  separate index publications alternating `index_files` and `index_v1`, then
  stats, deltalog, replacement, and deletion commits. Each commit clears the
  cache, reads persisted data, checks every remaining index, and decodes Avro
  directly to check stored paths against the pre-fix reader's resolution rule.
- `ManifestTest.IndexPathRepresentationsRoundTrip`: repeated reads/writes of
  plain, `./`, `../`, external-layout, empty, absolute, URI, prefix-lookalike,
  and lexically normalized inputs, including Manifest copies.
- `ManifestTest.IndexPathEditsAndSegmentCopy`: copying a read index, changing
  its path, copying the stored manifest into a different segment, and writing
  a loaded index to another segment while retaining its target.
- `ManifestTest.HistoricalIncorrectIndexPathIsNotGuessed`: historical incorrect
  paths stay incorrect and stable; no heuristic data repair occurs.

Validation results and commands are recorded below. A local
SubTreeFileSystem regression is distinct from a MinIO backend test. Neither is
Milvus integration, CI, or UAT evidence. Milvus's dependency pin and its
different-ID FFI/DataCoord regression should be updated after this fix is
published and merged; this change does not preemptively change that pin.

### Baseline reproduction (2026-10-19)

Built the unchanged implementation at the revision above with the first three
new tests. The transaction matrix and representation roundtrip failed; the
path-edit/file-copy test passed (3 tests, 2 failed, exit 1). The transaction
failure includes both synchronous and asynchronous calls and all three roots.
For example, the expected `index_files/0/1/P/S` became
`insert_log/C/P/sync/_index/index_files/0/1/P/S`; raw Avro contained the full
object key instead of `../../../../../index_files/0/1/P/S`.

All compilation and execution used the configured milvus-storage development
container via `wt-build`, preserving ASAN, CRT, unit tests, and fault injection.
The existing coverage files emitted checksum mismatch warnings after the
branch update; subsequent runs use separate data-backed coverage directories.

### Fixed implementation validation (2026-10-19)

| Scope | Result |
| --- | --- |
| C++ build, ASAN/CRT/FIU enabled | Passed |
| `ManifestTest.*:TransactionTest.*:AsyncTransactionTest.*`, local filesystem | 67 passed; 1 remote-only conditional-write test skipped; 0 failed |
| `Test_FFI`, local environment | 100 reported tests, 0 failed; cloud-only cases retain their existing skip behavior |
| MinIO: four added regressions plus concurrent conditional commits | 5 passed, 0 skipped, 0 failed |
| clang-format 18.1.8 on changed lines, `git diff --check` | Passed |
| Milvus Go/FFI/DataCoord integration, CI, UAT | Not run |

The MinIO run used the locally available `RELEASE.2025-09-07T16-13-09Z`
image, digest `sha256:69b2ec208575b69597784255eec6fa6a2985ee9e1a47f4411a51f7f5fdd193a9`,
on `localhost:19040`, with a dedicated bucket. It is a real local S3-compatible
backend test, not an AWS S3 or UAT test. Its container root was read-only,
file logging disabled, and data/tmp directories bind-mounted from `/data/yuruiz`.

Build through the configured local launcher:

```bash
wt-build <checkout> -- storage-build USE_ASAN=True WITH_CRT=True WITH_BENCHMARK=False
```

Inside that container, after sourcing `cpp/build/Release/generators/conanrun.sh`:

```bash
export GCOV_PREFIX="$BUILD_CACHE_LOG_DIR/gcov"
cpp/build/Release/test/milvus_test \
  --gtest_filter='ManifestTest.*:TransactionTest.*:AsyncTransactionTest.*'
cpp/build/Release/test/Test_FFI

env TEST_ENV_STORAGE_TYPE=remote TEST_ENV_ADDRESS=http://localhost:19040 \
  TEST_ENV_BUCKET_NAME=manifest-index-roundtrip TEST_ENV_REGION=us-east-1 \
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  cpp/build/Release/test/milvus_test \
  --gtest_filter='TransactionTest.IndexPathsSurviveIndependentCommits:TransactionTest.ConcurrentCommitsWithConditionalWrite:ManifestTest.IndexPath*:ManifestTest.HistoricalIncorrectIndexPathIsNotGuessed'
```

Local evidence is in the launcher's worktree log directory
`/data/yuruiz/build-cache/state/f7db44d3c4dbc13e-milvus-storage/worktrees/eacb77401875b444/logs/`:

- Baseline failure: `20261019T083543.646751Z`.
- Final build: `20261019T084042.440459Z`.
- Local C++ and FFI: `20261019T084245.094470Z`.
- MinIO: `20261019T084309.997767Z`.
- Changed-line format check: `20261019T084325.347161Z`.

Each directory contains `output.log` and `report.json`. The delivery commit
identifies this implementation; it must be published/merged before Milvus can
pin a remotely available revision.
