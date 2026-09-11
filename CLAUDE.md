# Milvus Storage

Columnar storage engine for Milvus, built on Apache Arrow. C++20 core (`cpp/`) with
Python, Java/Scala (JNI) and Rust bindings layered on a stable C ABI.

The C++ core is the only place logic lives. Every binding is a thin shim over
`extern "C"` entry points in `cpp/src/ffi/` — when you change behavior, change it in
the core and check whether the C ABI surface (`cpp/include/milvus-storage/ffi_*.h`)
needs to follow.

## Layout

| Path | Contents |
|------|----------|
| `cpp/include/milvus-storage/` | Public headers. `reader.h`, `writer.h`, `manifest.h`, `properties.h`, `column_groups.h`, and the `ffi_*_c.h` C ABI headers |
| `cpp/src/` | Implementation, mirroring the include tree |
| `cpp/test/` | GoogleTest suites (`milvus_test`, `Test_FFI`) |
| `cpp/benchmark/` | Google Benchmark suites — see `cpp/benchmark/benchmark.md` |
| `cpp/scripts/` | minio/azurite harnesses, clang-tidy driver, error-handling ratchet |
| `python/` | `milvus_storage` package (ctypes FFI + PyArrow), managed with `uv` |
| `java/` | Scala/sbt build over the JNI library |
| `rust/` | DataFusion `TableProvider` over the C ABI |
| `tests/` | Cross-language pytest integration + stress suites |
| `docs/` | Design documents (index below) |

### Core subsystems

Layers, top to bottom. `cpp/src/<name>/` and `cpp/include/milvus-storage/<name>/` pair up.

- **`ffi/`** — `extern "C"` boundary. Everything returns `LoonFFIResult{code, message}`;
  use the `RETURN_*` macros in `ffi_internal/result.h`, never raw returns.
  In the Python-binding build (`USE_PYTHON_BINDING=True`) visibility is hidden and only
  the symbols listed in `cpp/ffi_exports.map` (`ffi_exports_mac.map` on macOS) are
  exported — a new C entry point must be added there or it will not link from Python.
- **`transaction/`** — `Transaction::Open` → chained `Add*`/`Drop*` mutations → `Commit()`.
  Optimistic concurrency over manifest versions with a pluggable `Resolver` for conflicts.
- **`manifest.h` / `common/metadata.h`** — versioned table metadata: column groups, delta
  logs, indexes, statistics, LOB files. Manifests are cached (`common/lrucache.h`) and
  carry a magic number + format version; legacy versions have separate deserialize paths.
- **`segment/`** — `SegmentReader` / `SegmentWriter`: the row-level API, resolves
  `LOBReference` columns via `lob_column/` and supports sequential read plus `Take`.
- **`packed/`** — column-group packing: buffer management, row-group sizing, file rolling,
  chunk management, column projection. `splitter/` implements the column-group policies
  (single / schema-based / size-based) selected by `writer.policy`.
- **`format/`** — pluggable file formats: `parquet/` (primary), `vortex/`, `lance/`
  (read-only), `iceberg/`, `paimon/`. `bridge/` is the C++/Rust bridge those formats reach
  through; its sources are excluded from the main `milvus-storage` glob and linked in as
  the separate `prsbridge` target.
- **`filesystem/`** — Arrow `FileSystem` construction and credential handling per provider
  (`s3/`, `gcp/`, `azure/`, plus local). Cloud errors are classified here — see below.

### Design docs

Read the relevant one before changing the subsystem it covers; they record decisions the
code does not explain.

- [async-read-design.md](docs/async-read-design.md) — async open/read path, Folly executors
- [MANIFEST_EXTENSION_PLAN.md](docs/MANIFEST_EXTENSION_PLAN.md) — manifest format: delta logs, stats, versioning
- [manifest-index-artifacts-design.md](docs/manifest-index-artifacts-design.md) — publishing index artifacts through transactions
- [writer-policy-local-formats-design.md](docs/writer-policy-local-formats-design.md) — `writer.format` / per-group format selection
- [vortex-v2-row-group-zonemap-layout-plan.md](docs/vortex-v2-row-group-zonemap-layout-plan.md) — Vortex row-group zonemap layout
- [endpoint-use-ssl-design.md](docs/endpoint-use-ssl-design.md) — `fs.use_ssl` → endpoint scheme for Lance/Iceberg
- [gcp-cross-tenant-impersonation-design.md](docs/gcp-cross-tenant-impersonation-design.md) — GCP service-account impersonation
- [fat-jar-dependencies-plan.md](docs/fat-jar-dependencies-plan.md) — JNI fat-jar packaging and RUNPATH patching
- [integration-test-design.md](docs/integration-test-design.md) — `tests/` pytest suite design
- [multi-format-benchmark-design.md](docs/multi-format-benchmark-design.md), [predicate-pushdown-benchmark.md](docs/predicate-pushdown-benchmark.md) — benchmark design and results

## Build

Conan 2 + CMake, C++20, driven through `cpp/Makefile`. All commands run from `cpp/`.

```bash
conan profile detect --force
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2

make build                                  # defaults: Release, ASAN on, UT on, FIU on, benchmark on
make build BUILD_TYPE=Debug WITH_UT=False
make python-lib                             # shared lib for the Python FFI (hidden symbols, no ASAN)
make java-lib                               # libmilvus-storage-jni.so
make clean
```

Build options are Makefile **env vars**, not CMake flags:
`BUILD_TYPE`, `USE_ASAN`, `WITH_UT`, `WITH_FIU`, `WITH_CRT`, `WITH_BENCHMARK`, `USE_JNI`,
`USE_PYTHON_BINDING`. (The option table in `README.md` uses older `WITH_JNI=ON` style names —
the Makefile variables above are the ones that take effect.)

**Toggling a build option does not reconfigure an existing `build/` directory.** Changing
`USE_ASAN`, `WITH_UT`, etc. against a populated `build/` silently keeps the old
configuration. Run `make clean` first whenever you flip an option.

### Reuse the Milvus builder container

Build C++ artifacts inside the Milvus builder container rather than a storage-specific
image: it keeps the toolchain aligned with Milvus and shares its Conan 2 cache, so common
third-party dependencies are not downloaded twice. From the Milvus repository:

```sh
docker compose run --rm --no-deps \
  -v /abs/path/to/milvus-storage:/workspace/milvus-storage \
  -w /workspace/milvus-storage/cpp \
  builder sh -c 'make build; rc=$?; chown -R 1002:1002 /workspace/milvus-storage/cpp/build; exit $rc'
```

- Run the builder as root — its Rust toolchain lives under `/root/.cargo`.
- Always hand `cpp/build` back to the host UID afterwards, on success *and* failure.
- Do not `docker build` a separate storage image just to compile C++. The Java Compose
  files here target the JNI environment and use their own named Conan volume, so they do
  not share Milvus's cache.
- Building Milvus and milvus-storage concurrently is safe only if Conan cache access is
  coordinated; prefer the same builder and cache.

On a bare host, `libaio-dev` must be installed — Conan 2's CMakeDeps does not propagate
`system_libs` through shared targets, and folly needs it.

## Test

```bash
cd cpp
make test                       # milvus_test + Test_FFI, local filesystem only
make test-all                   # the above, then again against minio and azurite
make test-cloud-storage         # all providers; or: make test-cloud-storage aws
./build/Release/test/milvus_test --gtest_filter=Foo.Bar
./build/Release/benchmark/benchmark --benchmark_filter="Typical/"
```

`make test`/`make test-all` depend on `build`, so they rebuild first. To run a single
existing binary, invoke it directly as above.

```bash
cd python && make install && make lint && make test        # uv-managed; needs cpp `make python-lib`
cd rust  && cargo test && make lint                        # lint = clippy -D warnings
cd cpp && make java-lib && cd ../java && sbt compile       # JDK 21 + sbt
```

Cross-language integration and stress suites live in `tests/` and need both the FFI
library and the installed Python package:

```bash
cd cpp && make python-lib && cd ../python && pip install -e ".[dev]" && cd ..
pip install -r tests/requirements.txt
cd tests && pytest integration/ -v
cd tests && pytest stress/ --stress-scale=0.01 -v
```

## Error handling — machine-enforced

Library code reports failures as `arrow::Status` / `arrow::Result`. It must not abort or
throw across the library boundary. This is enforced by a CI ratchet, not by review alone.

```bash
cd cpp
make check-error-ratchet     # or: bash scripts/error_handling_ratchet.sh check
make update-error-ratchet    # only to record a burn-down
```

`cpp/scripts/error_handling_baseline.tsv` grandfathers the remaining `throw` sites under
`cpp/src` and `cpp/include`. Two layers guard it:

1. The committed baseline must match the tree **exactly**. A count that goes *down* means
   you must run `update` and commit the regenerated baseline in the same PR.
2. In CI, per-category totals are compared against the **base branch's** baseline. A count
   that goes *up* cannot be fixed by regenerating the baseline in your PR — that fails
   layer 2. Return a `Status` instead.

`cpp/test` is out of scope; comments are stripped before counting.

### Classification

`common/extend_status.h` defines `ExtendStatusCode` — codes ≥ 50, layered on top of
`arrow::StatusCode`, attached to a Status as an `ExtendStatusDetail`. The values are the
same integers as the `LOON_*` constants in `ffi_internal/ffi_error_code.h`, so a code set
deep in the filesystem layer survives all the way to the C ABI caller.

- Use `MakeExtendError(code, msg, extra)` to originate and `WrapExtendError(code, msg, cause)`
  to add context. Do not stringify a cause into a new message — that destroys the code.
- Transient (`StorageTransientNetwork/Timeout/Throttling/Service`, `AwsErrorConflict`) vs
  permanent (`AwsErrorNotFound`, `AwsErrorAccessDenied`, `AwsErrorNonRetryable`) is a
  **retry contract**, not a cosmetic label. Getting it backwards makes a caller either
  hammer a dead bucket or give up on a throttle.
- At the FFI boundary use `RETURN_ARROW_ERROR_IF(status, fallback, ...)`; the fallback only
  applies when the Status carries no `ExtendStatusDetail`. A boundary translator is only as
  correct as its input — audit the construction sites, not the translator.

## Verification gate

Before claiming a behavioral change works. "It compiles, unit tests pass, happy-path
integration is green" is evidence you did not break the happy path — nothing more.

**G1 — verify the input, not just your transform.** If your change maps or preserves a
value X (an error code, a retry flag, a format tag), auditing your own function proves
nothing. Grep every site across the repo that constructs, rewrites, or collapses X. A
catch-all that rewrites a classified error as `PackedUnexpected`, or a `ToString()` that
flattens a Status into a message, makes your boundary logic dead code.

**G2 — trace each real failure mode end to end.** For a change that exists to handle S3
throttling, a corrupt footer, a commit conflict, a cancelled read: either trace it by hand
from origin to consumer, or fault-inject it (`WITH_FIU=True`, `ffi_fiu_c.h`, `common/fiu_local.h`),
and confirm it lands in the intended bucket. Object-store failure modes are also reachable
via `make test-all` (minio/azurite) and `make test-cloud-storage`.

**G3 — do not over-claim.** Commit and PR text may assert only what G1+G2 verified.
An unverified benefit is written as a follow-up, not as achieved.

**G4 — adversarial pass before review.** Which failure mode did I not trace? Which
construction site of X did I not read? What would a reviewer grep for?

## Conventions

- **Formatting**: clang-format 18 (CI pins the version and will fail on any other).
  `make fix-format` from the repo root covers `cpp/` and `python/`; `make fix-checks` also
  runs clang-tidy.
- **clang-tidy**: `cd cpp && make check-tidy` / `make fix-tidy` — needs a configured
  `build/${BUILD_TYPE}` for `compile_commands.json`. Checks are in `cpp/.clang-tidy`.
- **Logging**: `common/log.h` macros only — `LOG_STORAGE_INFO_`, `LOG_STORAGE_ERROR_`,
  `LOG_STORAGE_DEBUG_`, etc. (glog underneath, with a `[STORAGE][func][thread]` prefix).
  No `std::cout`, no bare `LOG()`.
- **Configuration**: everything is a string-keyed property. A new key needs a
  `PROPERTY_*` macro in `properties.h`, a `REGISTER_PROPERTY(...)` entry in
  `properties.cpp` (type, description, default, validator), and — if callers set it
  across the ABI — a `loon_properties_*` constant in `ffi/properties_c.cpp`. Prefixes are
  meaningful (`fs.`, `extfs.`, `writer.`, `reader.`). Do not read environment variables
  from library code.
- **Naming**: `milvus_storage` is the root namespace; the current public API lives in
  `milvus_storage::api`, transactions in `milvus_storage::api::transaction`, segment-level
  code in `milvus_storage::segment`. `PackedRecordBatchReader` and friends in
  `milvus_storage` are the older packed-file API — prefer `api::Reader`/`api::Writer` for
  new code.
- **Rust bridge**: `cpp/src/format/bridge/rust/` is a Corrosion-driven cargo crate added
  as a CMake subdirectory and always built, so a stable Rust toolchain is required even
  for a plain C++ build. Changing its `Cargo.toml`/`Cargo.lock` invalidates the CI rust
  cache key.

## PR and commit conventions

PR title: `{type}: {description}`, matching the milvus-io convention —
`feat:`, `fix:`, `enhance:`, `test:`, `doc:`, `chore:`, `build(deps):`.
`fix:` and `feat:` should link an issue (`issue: #123`).

DCO is required. Always commit with `-s` so the developer's `Signed-off-by` is appended
last — the human, not the AI, must be the final sign-off:

```
git commit -s -m "enhance: <description>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Approvers/reviewers are listed in `OWNERS`.

## Do not hand-edit

- `cpp/build/` — generated; `make clean` to reset.
- `cpp/scripts/error_handling_baseline.tsv` — regenerate with `make update-error-ratchet`.
- `python/uv.lock`, `rust/Cargo.lock` — update through `uv` / `cargo`.
- `java/libmilvus-storage-jni.so` and `java/native/` — produced by `make java-lib` and the
  packaging scripts.
