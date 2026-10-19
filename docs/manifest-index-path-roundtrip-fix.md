# Require segment-local index paths at manifest commit

Index metadata may only reference artifacts beneath the current segment's
`_index/` directory. `Manifest::WriteTo` validates every retained and newly
added index before serializing or opening an output stream. An invalid path
returns an error and leaves the next manifest version unpublished.

## Path contract

Writers must supply a full, canonical filesystem path or object key with a
nonempty suffix after `<segment>/_index/`. For example, for segment
`R/insert_log/C/P/S`:

| Input | Result |
| --- | --- |
| `R/insert_log/C/P/S/_index/build-A` | Accepted; stored as `build-A` |
| `R/insert_log/C/P/S/_index/build-A/` | Accepted; directory suffix retained |
| `build-A` or `./build-A` | Rejected; missing segment prefix |
| `../../../../../index_files/A` | Rejected |
| `R/index_files/A` or `R/index_v1/A` | Rejected |
| `R/insert_log/C/P/S/_index-other/A` | Rejected; not an `_index/` child |
| `R/insert_log/C/P/S/_index/../outside` | Rejected |
| `_index` directory itself, empty path, external URI | Rejected |

The prefix includes the directory separator. Paths with redundant separators,
`.` or `..` components are rejected rather than normalized and redirected.
Local absolute paths are accepted when they use the current absolute segment
prefix. This is a lexical metadata check; it does not issue object-existence
requests or resolve filesystem symlinks.

Writing strips the accepted prefix using the existing conversion. Reading
restores it using the existing reader. No path provenance is retained and no
`Index` layout or manifest schema change is needed. Column groups, stats,
deltalogs, LOB paths, index identities, and conflict resolution are unchanged.
Raw backup/restore `serialize`/`deserialize` codecs remain raw codecs; without
a segment context they do not perform this commit-time check.

## Transaction behavior and callers

`AddIndexInfo` stages metadata; validation happens at commit, after updates
have been applied. Both synchronous and asynchronous commits call the same
`Manifest::WriteTo` implementation. Validation covers old indexes retained by
stats, deltalog, or unrelated index updates as well as newly added indexes.
Dropping an invalid old index can therefore unblock future commits.

This intentionally removes support for publishing legacy `index_files` and
external `index_v1` layouts. Milvus callers that currently generate `../...`
paths must instead produce artifacts beneath the segment's `_index/` and pass
the full directory to the transaction API before adopting this change. Bare
relative paths in existing FFI examples/tests have been updated to full paths.
No Milvus dependency pin, file placement, or UAT data is changed here.

Reading old manifests remains unchanged. A legacy index expanded outside
`_index/` prevents any commit that retains it. Already-corrupted metadata that
looks like a path *inside* `_index/` cannot be identified by prefix validation;
this check does not repair or prove the correctness of historical records.

Copying stored Avro to a new segment retains the existing relative-path
rebasing behavior. Writing an already-expanded manifest from another segment
fails unless callers update its index paths to the new segment's prefix.

## Regression coverage

- Ten independent indexes followed by stats, deltalog, replacement, and
  deletion commits; sync and async entrypoints; empty, single-component, and
  multi-component roots. Every commit clears the cache and verifies all index
  metadata and stored Avro suffixes.
- Invalid prefixes, legacy layouts, relative inputs, URIs, empty paths, and
  traversal/noncanonical inputs: errors leave no new manifest and do not
  publish an output size.
- New invalid indexes and retained legacy indexes both fail through sync/async
  commits; reopening sees the original version and dropping the invalid index
  allows a new commit.
- Local absolute paths and copying stored manifests to another segment.

All builds, tests, and formatting run in the configured development container
through `wt-build`, with ASAN, CRT, and FIU enabled and data-backed caches and
outputs. Verification results for this revised policy are recorded below;
results for the preceding compatibility implementation do not validate it.

### Verification of the revised policy

Before adding validation, both new rejection tests failed against `0f5022e4`:
invalid paths were committed and the next manifest version was visible.
The baseline build/run exited 1; its log is
`20261019T085215.714748Z/output.log` under this worktree's launcher log directory.

After validation was added:

| Check | Result |
| --- | --- |
| C++ build with ASAN/CRT/FIU | Passed |
| `ManifestTest.*:TransactionTest.*:AsyncTransactionTest.*` | 69 passed, 1 remote-only test skipped, 0 failed |
| Local `Test_FFI` | 100 reported tests, 0 failed; existing cloud-only skip retained |
| MinIO: valid roundtrips, invalid/new/retained paths, copying, conditional commits | 6 passed, 0 skipped, 0 failed |
| clang-format 18.1.8 on changed lines; `git diff --check` | Passed |

Build and local tests used:

```bash
wt-build <checkout> -- bash -c '
  cmake --build cpp/build/Release --target milvus_test Test_FFI -j 32 &&
  source cpp/build/Release/generators/conanrun.sh &&
  export GCOV_PREFIX="$BUILD_CACHE_LOG_DIR/gcov" &&
  cpp/build/Release/test/milvus_test --gtest_filter=ManifestTest.*:TransactionTest.*:AsyncTransactionTest.* &&
  cpp/build/Release/test/Test_FFI'
```

The MinIO run used `TEST_ENV_STORAGE_TYPE=remote`, address
`http://localhost:19040`, bucket `manifest-index-roundtrip`, region `us-east-1`,
and direct localhost proxy exclusions. It ran the two new transaction tests,
the existing concurrent-commit test, `ManifestTest.IndexPaths*`, and
`ManifestTest.RejectInvalidIndexPathsBeforeWriting`. The server was
`RELEASE.2025-09-07T16-13-09Z`, with a read-only container root, logging disabled,
and data-backed bind mounts. This is local S3-compatible backend evidence;
Milvus integration, CI, AWS S3, and UAT were not run.

Evidence under
`/data/yuruiz/build-cache/state/f7db44d3c4dbc13e-milvus-storage/worktrees/eacb77401875b444/logs/`:

- Baseline failure: `20261019T085215.714748Z`.
- Build, local C++ and FFI: `20261019T085423.905526Z`.
- MinIO: `20261019T085732.007875Z`.
- Changed-line format check: `20261019T085747.218670Z`.

Each run has `output.log` and `report.json`.
