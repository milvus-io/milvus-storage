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
./build/Release/benchmark/benchmark --benchmark_filter="Typical/"

# Multiple patterns (use | separator)
./build/Release/benchmark/benchmark --benchmark_filter="MilvusStorage_Read|MilvusStorage_Take"

# Exclude patterns (use - prefix)
./build/Release/benchmark/benchmark --benchmark_filter="-Typical/"
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

## Typical Benchmarks

Benchmarks with `Typical/` prefix are representative tests with common parameter configurations, suitable for quick validation:

```bash
./build/Release/benchmark/benchmark --benchmark_filter="Typical/"
```
