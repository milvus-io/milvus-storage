# Milvus Storage Benchmark

Benchmark suite for testing Milvus Storage read/write performance.

## Build

```bash
cd cpp
make build
```

The benchmark executable is located at `build/Release/benchmark/benchmark`.

## Run

```bash
# Run all benchmarks
./build/Release/benchmark/benchmark

# Filter by regex
./build/Release/benchmark/benchmark --benchmark_filter="FormatReadBenchmark/ReadFullScan"

# Multiple patterns (use | separator)
./build/Release/benchmark/benchmark --benchmark_filter="MilvusStorage_Read|MilvusStorage_Take"

# List matching benchmarks without running them
./build/Release/benchmark/benchmark --benchmark_filter="FormatReadBenchmark" --benchmark_list_tests=true
```

## Benchmark Files

### benchmark_format_read.cpp
**Format Layer Read Performance**

Tests read performance for different storage formats (Parquet, Vortex):

| Benchmark | Description | Args |
|-----------|-------------|------|
| `ReadFullScan` | Full table scan, read all rows and columns | format, num_threads, memory_config |
| `ReadProjection` | Column projection, read subset of columns | format, num_columns, num_threads, memory_config |
| `ReadTake` | Random access by indices | format, take_count, distribution, num_threads, memory_config |

**Format index**: 0=Parquet, 1=Vortex

### benchmark_format_write.cpp
**Format Layer Write Performance**

| Benchmark | Description | Args |
|-----------|-------------|------|
| `WriteComparison` | Write performance comparison | format, data_config, memory_config |
| `CompressionAnalysis` | Compression ratio analysis | format, data_config |

**Data config index**: 0=Small(rK rows), 1=Medium(40K rows), 2=Large(409K rows), 3=HighDim, 4=LongString

### benchmark_storage_layer.cpp
**Storage Layer End-to-End Performance (with Transaction)**

Compares MilvusStorage (Parquet/Vortex + Transaction) vs Lance Native:

| Benchmark | Description | Args |
|-----------|-------------|------|
| `MilvusStorage_WriteCommit` | Write + transaction commit | format_type, data_config |
| `MilvusStorage_WriteOnly` | Write only (no transaction) | format_type, data_config |
| `MilvusStorage_OpenRead` | Open transaction + read | format_type, data_config, num_threads |
| `MilvusStorage_Take` | Open transaction + take | format_type, take_count, num_threads |
| `MilvusStorage_MultiReader` | Multi-reader concurrency test | format_type, num_readers, thread_pool_size |
| `LanceNative_WriteCommit` | Lance write | data_config |
| `LanceNative_OpenRead` | Lance read | data_config, num_threads |
| `LanceNative_Take` | Lance take | take_count, num_threads |
| `LanceNative_MultiReader` | Lance multi-reader concurrency | num_readers, thread_pool_size |

**Format type index**: 0=Parquet, 1=Vortex, 2=Mixed(Parquet+Vortex)

**Note**: Lance benchmarks require `BUILD_LANCE_BRIDGE` enabled at compile time.

**TODO**: Lance S3 storage requires separate configuration (not using MilvusStorage's filesystem layer).

### benchmark_v2_v3.cpp
**V2 vs V3 API Performance Comparison**

Compares low-level Packed API (V2) vs high-level Reader/Writer API (V3):

| Benchmark | Description | Args |
|-----------|-------------|------|
| `V2_PackedRecordBatchReader` | Low-level PackedRecordBatchReader | data_config |
| `V2_PackedRecordBatchWriter` | Low-level PackedRecordBatchWriter | data_config |
| `V3_RecordBatchReader` | High-level Reader API (get_record_batch_reader) | data_config |
| `V3_ChunkReader` | High-level Reader API (get_chunk_reader) | data_config |
| `V3_Writer` | High-level Writer API | data_config |

### benchmark_footer_size.cpp
**Parquet Footer Size Analysis**

Measures Parquet file footer size and its percentage of total file size:

| Benchmark | Description | Args |
|-----------|-------------|------|
| `MeasureFooterSize` | Footer size measurement | num_rows, vector_dim, string_length |

Output metrics: `footer_size_bytes`, `file_size_bytes`, `footer_percentage`

### benchmark_wr.cpp
**Basic Read/Write Performance**

Simple read/write performance tests for quick validation:

| Benchmark | Description | Args |
|-----------|-------------|------|
| `WriteDefaultConfig` | Default config write | loop_times |
| `WriteSingleColumnConfig` | Single column write | loop_times, column_idx |
| `ReadFullScanDefaultConfig` | Default config full scan | loop_times |
| `ReadFullScanSingleColumnConfig` | Single column full scan | loop_times, column_idx |
| `WriteRead768dimVector` | 768-dim vector large file test | target_size, target_dim |

## Nightly CI Reader Benchmarks

`ReaderBenchmark/` is the canonical public-Reader benchmark family.
`NIGHTLY_CI_TARGET/` contains name-only aliases of the same configurations and
execution bodies. Each family contains 91 registrations with identical
semantic suffixes.

Run either family locally after configuring storage, or validate their exact
one-to-one registration contract:

```bash
./build/Release/benchmark/benchmark --benchmark_filter='^ReaderBenchmark/'
./build/Release/benchmark/benchmark --benchmark_filter='^NIGHTLY_CI_TARGET/'
NIGHTLY_CI_EXECUTOR_THREADS=3 ./build/Release/benchmark/benchmark \
  --benchmark_filter='^NIGHTLY_CI_TARGET/Async/'
bash benchmark/nightly/test_reader_alias.sh \
  ./build/Release/benchmark/benchmark registration
```

An unfiltered invocation executes both name families. Use an explicit prefix
filter unless that duplication is intentional.

Each family has 63 synchronous cases and 28 asynchronous cases (91 total).
Each applicable operation/format combination runs these seven deterministic
dataset scenarios:

| Dataset | Contents |
|---------|----------|
| `SyntheticSmall` | 4,096 rows with scalar columns and a random 128-dimensional vector |
| `SyntheticMedium` | 40,960 rows with scalar columns and a random 128-dimensional vector |
| `SyntheticLarge` | 409,600 rows with scalar columns and a random 128-dimensional vector |
| `ScalarMedium` | 40,960 scalar-only rows |
| `RandomVector64MiB` | Random 256-dimensional vector data with a 64 MiB raw payload |
| `LowEntropyVector256MiB` | Deterministic low-entropy 256-dimensional vector data with a 256 MiB raw payload |
| `RandomVector2GiB` | Random 256-dimensional vector data with a 2 GiB raw payload |

Synchronous cases cover record-batch reads, chunk reads, and `take` for
Parquet, Vortex, and Lance. The asynchronous cases are genuine public async
calls only: chunk reads and `take` for Parquet and Vortex. There is no public
async record-batch API, and Lance has no native async override, so neither is
reported as an async case. The `parallelism` API argument is intentionally
omitted, retaining its default value of `1`.

Async continuations run on a caller-owned `folly::CPUThreadPoolExecutor`.
Set `NIGHTLY_CI_EXECUTOR_THREADS` to a positive integer to size that executor;
when it is absent, the local default is `1`. This setting sizes the executor,
not process affinity. In hosted CI, MinIO is pinned to logical CPU `0` and the
complete benchmark process is pinned to logical CPUs `1,2,3`; the executor
thread count is derived from the benchmark CPU set. This is logical-CPU
separation, not physical-core isolation, and local smoke runs do not validate
the hosted affinity configuration.

The workflow writes raw Google Benchmark JSON and text output as diagnostic
artifacts. It also renders a human-readable Markdown summary with median
real-time, variation, throughput, and sync/async comparisons; this summary is
printed in the job log, added to the GitHub Step Summary, and stored alongside
the raw artifacts.
