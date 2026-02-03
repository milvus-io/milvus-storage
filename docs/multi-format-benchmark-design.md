# Multi-Format Benchmark Design

## Overview

This document outlines the design for adding multi-format benchmark tests to compare performance characteristics across different storage formats supported by Milvus Storage: **Parquet**, **Vortex**, and **Lance** (read-only).

## Current State

### Supported Formats

| Format | Write | Read | Status |
|--------|-------|------|--------|
| Parquet | Yes | Yes | Default format |
| Vortex | Yes | Yes | Optional |
| Lance | No | Yes | Read-only, Optional |

Format constants defined in `cpp/include/milvus-storage/common/config.h`:

```cpp
#define LOON_FORMAT_PARQUET "parquet"
#define LOON_FORMAT_VORTEX "vortex"
#define LOON_FORMAT_LANCE_TABLE "lance-table"
```

### Existing Benchmark Structure

```
cpp/benchmark/
├── benchmark_main.cpp          # Entry point
├── benchmark_wr.cpp            # Write/Read benchmarks (Parquet only)
├── benchmark_footer_size.cpp   # Parquet footer size analysis
└── CMakeLists.txt              # Build configuration
```

Current benchmarks only test Parquet format. No cross-format comparison exists.

### Key Helper Functions

From `test_env.h` / `test_env.cpp`:

```cpp
// Generate available formats based on build configuration
std::vector<std::string> GenerateFormatTestPValuesIn();

// Create policy with specific format
arrow::Result<std::unique_ptr<api::ColumnGroupPolicy>> CreateSinglePolicy(
    const std::string& format,
    const std::shared_ptr<arrow::Schema>& schema);

// Set format in properties
SetValue(properties, PROPERTY_FORMAT, format.c_str());
```

## Proposed Benchmarks

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Benchmark Hierarchy                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: Format Layer (Pure format read/write)                         │
│  ┌─────────────────┬─────────────────┐                                  │
│  │ Parquet Writer  │ Vortex Writer   │                                  │
│  │ Parquet Reader  │ Vortex Reader   │                                  │
│  └─────────────────┴─────────────────┘                                  │
│  Benchmarks: Write Performance, Read Performance, V2 vs V3 Reader       │
│                                                                         │
│  Level 2: Storage Layer (With Transaction)                              │
│  ┌─────────────────┬─────────────────┬─────────────────┐                │
│  │ Writer + Txn    │ Writer + Txn    │ Lance Native    │                │
│  │ (Parquet)       │ (Vortex)        │ (Built-in Txn)  │                │
│  │ Txn + Reader    │ Txn + Reader    │ Dataset.scan()  │                │
│  └─────────────────┴─────────────────┴─────────────────┘                │
│  Benchmark: Storage Layer Comparison                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Level 1: Format Layer Benchmarks

| Benchmark | File | Description |
|-----------|------|-------------|
| **Write Performance** | `benchmark_format_write.cpp` | Compare write throughput and file size/compression across formats |
| **Read Performance** | `benchmark_format_read.cpp` | Full scan, column projection, multi-threading, and random access (take) |
| **V2 vs V3 Reader** | `benchmark_v2_v3.cpp` | Compare packed/ low-level reader vs top-level API overhead |

##### Write Performance Benchmark (benchmark_format_write.cpp)

Measures and compares write performance between Parquet and Vortex formats:

| Capability | Description | Parameters |
|------------|-------------|------------|
| **Write Throughput** | Measure MB/s and rows/s for batch writes | Format, Data Size, Memory Config |
| **File Size Analysis** | Compare output file sizes across formats | Format, Data Type (random/sequential/sparse/repetitive) |
| **Compression Ratio** | Calculate compressed_size / raw_data_size | Format, Data Type |
| **Metadata Overhead** | Measure metadata size as percentage of total | Format, Data Size |
| **Memory Usage** | Track peak memory during write operations | Format, Buffer Size, Batch Size |
| **Data Size Scaling** | Test with Small (4K), Medium (40K), Large (400K) rows | Data Size Config |
| **Schema Variations** | Test HighDim vectors (768), LongString (1024 chars) | Schema Config |

##### Read Performance Benchmark (benchmark_format_read.cpp)

Comprehensive read performance testing with multiple scenarios:

| Capability | Description | Parameters |
|------------|-------------|------------|
| **Full Scan (ReadFullScan)** | Read all rows and all columns | Format, Memory Config |
| **Column Projection (ReadProjection)** | Read subset of columns (1/2/3/4 of 4) | Format, Num Columns, Memory Config |
| **Projection Efficiency** | Calculate speedup from skipping columns | Format, Num Columns |
| **Multi-threading (ReadParallel)** | Test with 1/2/4/8/auto threads | Format, Thread Count, Memory Config |
| **Parallel Speedup** | Calculate single_thread_time / n_thread_time | Format, Thread Count |
| **Parallel Efficiency** | Calculate speedup / n_threads | Format, Thread Count |
| **Random Access (ReadTake)** | Take specific rows by indices | Format, Take Count, Distribution, Memory Config |
| **Distribution Patterns** | Sequential, Random, Clustered index patterns | Distribution Type |
| **Memory Usage** | Track peak memory for all read operations | Memory Config |

##### V2 vs V3 Reader Benchmark (benchmark_v2_v3.cpp)

Compare low-level PackedRecordBatchReader (v2) vs top-level Reader API (v3):

| Capability | Description | Parameters |
|------------|-------------|------------|
| **V2 PackedRecordBatchReader** | Direct Parquet file reading (low-level) | Data Size Config |
| **V3 RecordBatchReader** | Top-level Reader::get_record_batch_reader() | Data Size Config |
| **V3 ChunkReader** | Top-level Reader::get_chunk_reader() | Data Size Config |
| **API Overhead** | Calculate (v3_time - v2_time) / v2_time | Data Size Config |
| **Multi-file Scaling** | Test with 1 file vs 10 files | File Count |

> **Note:** V2 vs V3 benchmark is Parquet-only since PackedRecordBatchReader is Parquet-specific.

#### Level 2: Storage Layer Benchmarks

| Benchmark | File | Description |
|-----------|------|-------------|
| **Storage Layer Comparison** | `benchmark_storage_layer.cpp` | Compare Milvus-Storage (Parquet/Vortex + Transaction) vs Lance Native (built-in transaction) |

##### Storage Layer Benchmark (benchmark_storage_layer.cpp)

End-to-end storage system comparison including Transaction layer:

| Capability | Description | Parameters |
|------------|-------------|------------|
| **Write + Commit (MilvusStorage)** | Writer::write() + Writer::close() + Transaction::Commit() | Format Type, Data Size Config |
| **Write + Commit (LanceNative)** | lance::write_dataset() with built-in commit | Data Size Config |
| **Open + Read (MilvusStorage)** | Transaction::Open(fs, path) + GetManifest() + Reader::get_record_batch_reader() | Format Type, Data Size Config |
| **Open + Read (LanceNative)** | lance::open_dataset() + scan() | Data Size Config |
| **Random Access (MilvusStorage)** | Transaction::Open(fs, path) + GetManifest() + Reader::take() | Format Type, Take Count |
| **Random Access (LanceNative)** | lance::open_dataset() + take() | Take Count |
| **Mix Write + Commit (MilvusStorage)** | Parquet (scalar cols) + Vortex (vector col) combined via Transaction | Mixed Config |
| **Mix Open + Read (MilvusStorage)** | Read from manifest with multiple format column groups | Mixed Config |

##### Format Types for Storage Layer

| Format Type | Value | Description |
|-------------|-------|-------------|
| PARQUET | 0 | All columns written with Parquet format |
| VORTEX | 1 | All columns written with Vortex format |
| MIXED | 2 | Scalar columns (id, name, value) with Parquet + Vector column with Vortex |

##### Mixed Format Flow

```
Write Flow:
1. Writer::create(path, scalar_schema, parquet_policy) → write scalar columns → close()
2. Writer::create(path, vector_schema, vortex_policy) → write vector column → close()
3. Transaction::Open(fs, path) → AddColumnGroup(scalar_cgs) → AddColumnGroup(vector_cgs) → Commit()

Read Flow:
1. Transaction::Open(fs, path) → get manifest with column group info
2. Reader::create(cgs, schema) → get_record_batch_reader() / take() (auto-handles mixed formats)
```

**Key Dimensions:**
- **Formats** (Level 1): Parquet, Vortex
- **Formats** (Level 2): Parquet, Vortex, Mixed (scalar:Parquet + vector:Vortex)
- **Data Sizes**: Small (4K rows), Medium (40K rows), Large (400K rows)
- **Storage Backend**: Local filesystem, Cloud storage (AWS S3, GCP, Azure, Aliyun, etc.)
- **Memory**: Low (16MB buffer), Default (64MB), High (256MB)

### 1. Write Performance Benchmark

**File:** `cpp/benchmark/benchmark_format_write.cpp`

**Purpose:** Compare write throughput across formats with identical data, and measure file size/compression efficiency.

#### 1.1 Test Configurations

| Config Name | Rows | Vector Dim | String Length | Description |
|-------------|------|------------|---------------|-------------|
| Small | 4,096 | 128 | 128 | Default workload |
| Medium | 40,960 | 128 | 128 | 10x rows |
| Large | 409,600 | 128 | 128 | 100x rows |
| HighDim | 4,096 | 768 | 128 | High-dimensional vectors |
| LongString | 4,096 | 128 | 1024 | Long string values |

**Metrics to Collect:**
- Time per batch write (ms)
- Total write throughput (MB/s)
- Rows per second
- File size (bytes)
- Compression ratio (file_size / raw_data_size)
- Metadata overhead (%)
- Peak memory usage (MB)
- Memory efficiency (MB/s per MB memory)

#### 1.2 Memory Configurations

Test write performance under different memory buffer settings:

| Memory Config | Write Buffer | Batch Size | Description |
|---------------|--------------|------------|-------------|
| Low | 16 MB | 1024 | Memory-constrained environment |
| Default | 128 MB | 16384 | Standard configuration |
| High | 256 MB | 32768 | Memory-rich environment |

#### 1.3 File Size / Compression Analysis

Measure file size and compression efficiency after write completion.

**Test Data Variations:**

| Data Type | Description |
|-----------|-------------|
| Random | Random values (low compressibility) |
| Sequential | Sequential values (high compressibility) |
| Sparse | Many null/zero values |
| Repetitive | Repeated patterns |

**Expected Output:**

```
Format     | DataType   | RawSize | FileSize | Ratio  | Overhead
-----------|------------|---------|----------|--------|----------
parquet    | random     | 10 MB   | 8.5 MB   | 0.85   | 2.1%
vortex     | random     | 10 MB   | 7.2 MB   | 0.72   | 1.8%
parquet    | sequential | 10 MB   | 2.1 MB   | 0.21   | 5.2%
vortex     | sequential | 10 MB   | 1.8 MB   | 0.18   | 4.1%
```

#### 1.4 Benchmark Definition

```cpp
struct FormatWriteConfig {
  std::string format;           // "parquet", "vortex"
  size_t num_rows = 4096;
  size_t vector_dim = 128;
  size_t string_length = 128;
  size_t loop_times = 10;
  size_t write_buffer_size = 64 * 1024 * 1024;  // 64 MB default
  size_t batch_size = 8192;
};

BENCHMARK_DEFINE_F(FormatBenchFixture, WriteComparison)(benchmark::State& st) {
  std::string format = formats[st.range(0)];  // 0=parquet, 1=vortex
  size_t config_idx = st.range(1);            // Config index
  size_t memory_config = st.range(2);         // 0=low, 1=default, 2=high

  // Configure memory settings
  size_t buffer_sizes[] = {16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024};
  size_t batch_sizes[] = {1024, 8192, 32768};
  SetValue(properties_, PROPERTY_WRITER_BUFFER_SIZE, std::to_string(buffer_sizes[memory_config]));
  SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, std::to_string(batch_sizes[memory_config]));

  // Track memory
  auto* pool = arrow::default_memory_pool();
  int64_t initial_memory = pool->bytes_allocated();

  for (auto _ : st) {
    // ... write implementation
  }

  // Report memory metrics
  st.counters["peak_memory_mb"] = (pool->max_memory() - initial_memory) / (1024.0 * 1024.0);
}

BENCHMARK_REGISTER_F(FormatBenchFixture, WriteComparison)
    ->ArgsProduct({
        {0, 1},           // Format: parquet, vortex
        {0, 1, 2, 3, 4},  // Config: Small, Medium, Large, HighDim, LongString
        {0, 1, 2}         // Memory: Low, Default, High
    })
    ->Unit(benchmark::kMillisecond);
```

### 2. Read Performance Benchmark (includes Column Projection & Multi-threading)

**File:** `cpp/benchmark/benchmark_format_read.cpp`

**Purpose:** Compare read performance across formats, including full-scan, column projection, and multi-threading scenarios.

#### 2.1 Test Data Configuration

Read benchmarks use the same data size configurations as write benchmarks:

| Config Name | Rows | Vector Dim | String Length | Description |
|-------------|------|------------|---------------|-------------|
| Small | 4,096 | 128 | 128 | Default workload |
| Medium | 40,960 | 128 | 128 | 10x rows |
| Large | 409,600 | 128 | 128 | 100x rows |

#### 2.2 Memory Configurations

Test read performance under different memory buffer settings:

| Memory Config | Read Buffer | Batch Size | Description |
|---------------|-------------|------------|-------------|
| Low | 16 MB | 1024 | Memory-constrained environment |
| Default | 128 MB | 16384 | Standard configuration |
| High | 256 MB | 32768 | Memory-rich environment |

#### 2.3 Full Scan Scenarios

| Scenario | Columns | Description |
|----------|---------|-------------|
| FullScan | all 4 | Read all columns (id, name, value, vector) |
| VectorOnly | vector | Read only vector column (large column) |
| ScalarOnly | id, name, value | Read only scalar columns |

#### 2.4 Column Projection Scenarios

Compare column projection performance across formats, measuring the efficiency of skipping unnecessary columns.

| Scenario | Projection | Columns Read | Description |
|----------|------------|--------------|-------------|
| Proj_1_of_4 | 25% | 1 column | Read only 1 column |
| Proj_2_of_4 | 50% | 2 columns | Read only 2 columns |
| Proj_3_of_4 | 75% | 3 columns | Read only 3 columns |
| Proj_4_of_4 | 100% | 4 columns | Read all columns (baseline) |

**Projection Efficiency Metric:**

```
Projection Efficiency = (FullScan Time) / (Projection Time)

Ideal: Reading 25% of columns takes ~25% time, efficiency = 4.0x
Actual: Due to metadata overhead, efficiency < 4.0x
```

#### 2.5 Multi-threading Scenarios

Test the impact of different thread counts on read performance using `ThreadPoolHolder` to configure the thread pool.

| Threads | Description |
|---------|-------------|
| 1 | Single thread (baseline) |
| 2 | 2 threads |
| 4 | 4 threads |
| 8 | 8 threads |
| 0 | Auto (CPU core count) |

**Parallel Efficiency Metric:**

```
Parallel Speedup = (Single Thread Time) / (N Threads Time)
Parallel Efficiency = Speedup / N

Ideal: 4 threads speedup = 4.0x, efficiency = 100%
Actual: Due to I/O bottleneck, lock contention, etc., efficiency < 100%
```

**Thread Pool Configuration:**

```cpp
// Configure thread pool in benchmark SetUp
ThreadPoolHolder::WithSingleton(num_threads);

// Release in benchmark TearDown
ThreadPoolHolder::Release();
```

#### 2.6 Random Access (take) Scenarios

Test `Reader::take()` random access performance, which is important for vector search TopK result retrieval.

**Take Count Configurations:**

| Scenario | Take Count | Description |
|----------|------------|-------------|
| Take_10 | 10 rows | Small batch random access |
| Take_100 | 100 rows | Medium batch random access |
| Take_1000 | 1000 rows | Large batch random access |
| Take_10000 | 10000 rows | Extra large batch random access |

**Row Index Distribution Patterns:**

| Distribution | Description | Use Case |
|--------------|-------------|----------|
| Sequential | Contiguous row indices (e.g., 100-199) | Range queries, pagination |
| Random | Uniformly distributed random indices | Typical vector search results |
| Clustered | Multiple small clusters (e.g., 10-15, 100-105, 500-510) | Filtered search results |

**Distribution Generation:**

```cpp
// Sequential: indices are contiguous
std::vector<int64_t> GenerateSequentialIndices(size_t count, int64_t start) {
  std::vector<int64_t> indices(count);
  std::iota(indices.begin(), indices.end(), start);
  return indices;
}

// Random: uniformly distributed
std::vector<int64_t> GenerateRandomIndices(size_t count, int64_t max_value) {
  std::vector<int64_t> indices;
  std::set<int64_t> seen;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dist(0, max_value - 1);
  while (indices.size() < count) {
    int64_t idx = dist(gen);
    if (seen.insert(idx).second) {
      indices.push_back(idx);
    }
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

// Clustered: multiple small contiguous clusters
std::vector<int64_t> GenerateClusteredIndices(size_t count, int64_t max_value, size_t cluster_size = 5) {
  std::vector<int64_t> indices;
  size_t num_clusters = (count + cluster_size - 1) / cluster_size;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dist(0, max_value - cluster_size);
  while (indices.size() < count) {
    int64_t start = dist(gen);
    for (size_t j = 0; j < cluster_size && indices.size() < count; ++j) {
      indices.push_back(start + j);
    }
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}
```

**Random Access Efficiency Metric:**

```
Random Access Efficiency = (Sequential Scan Time for N rows) / (Take N rows Time)

Higher value indicates better random access efficiency compared to sequential scan
```

**Benchmark Registration:**

```cpp
// Distribution: 0=sequential, 1=random, 2=clustered
BENCHMARK_REGISTER_F(FormatBenchFixture, ReadTake)
    ->ArgsProduct({
        {0, 1},                    // Format: parquet, vortex
        {10, 100, 1000, 10000},    // Number of rows to take
        {0, 1, 2}                  // Distribution: sequential, random, clustered
    })
    ->Unit(benchmark::kMillisecond);
```

#### 2.7 Metrics to Collect

- Time to read data (ms)
- Read throughput (MB/s)
- Rows per second
- Projection efficiency (full_scan_time / projection_time)
- Parallel speedup (single_thread_time / n_thread_time)
- Parallel efficiency (speedup / n_threads)
- Random access latency (ms per row)
- Peak memory usage (MB)
- Memory efficiency (MB/s per MB memory)

#### 2.8 Benchmark Definition

```cpp
struct FormatReadConfig {
  std::string format;
  std::array<bool, 4> projection;  // {id, name, value, vector}
  size_t num_threads;
  size_t read_buffer_size = 64 * 1024 * 1024;  // 64 MB default
  size_t batch_size = 8192;
  std::string scenario_name;
};

// Memory configuration helper
void ConfigureMemory(api::Properties& props, size_t memory_config) {
  size_t buffer_sizes[] = {16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024};
  size_t batch_sizes[] = {1024, 8192, 32768};
  SetValue(props, PROPERTY_WRITER_BUFFER_SIZE, std::to_string(buffer_sizes[memory_config]));
  SetValue(props, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, std::to_string(batch_sizes[memory_config]));
}

// Full scan benchmark with memory configuration
BENCHMARK_DEFINE_F(FormatBenchFixture, ReadFullScan)(benchmark::State& st) {
  std::string format = formats[st.range(0)];
  size_t memory_config = st.range(1);  // 0=low, 1=default, 2=high

  ConfigureMemory(properties_, memory_config);
  ThreadPoolHolder::WithSingleton(1);

  auto* pool = arrow::default_memory_pool();
  int64_t initial_memory = pool->bytes_allocated();

  for (auto _ : st) {
    // Read all columns
  }

  st.counters["peak_memory_mb"] = (pool->max_memory() - initial_memory) / (1024.0 * 1024.0);
}

// Column projection benchmark with memory configuration
BENCHMARK_DEFINE_F(FormatBenchFixture, ReadProjection)(benchmark::State& st) {
  std::string format = formats[st.range(0)];
  size_t num_columns = st.range(1);  // 1, 2, 3, or 4 columns
  size_t memory_config = st.range(2);

  ConfigureMemory(properties_, memory_config);
  ThreadPoolHolder::WithSingleton(1);
  // Read subset of columns, measure projection efficiency
}

// Multi-threading benchmark with memory configuration
BENCHMARK_DEFINE_F(FormatBenchFixture, ReadParallel)(benchmark::State& st) {
  std::string format = formats[st.range(0)];
  size_t num_threads = st.range(1);  // 1, 2, 4, 8, or 0 (auto)
  size_t memory_config = st.range(2);

  ConfigureMemory(properties_, memory_config);

  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  ThreadPoolHolder::WithSingleton(num_threads);

  auto* pool = arrow::default_memory_pool();
  int64_t initial_memory = pool->bytes_allocated();

  for (auto _ : st) {
    // ... read operation
  }

  // Report metrics
  st.counters["threads"] = num_threads;
  st.counters["speedup"] = baseline_time / actual_time;
  st.counters["peak_memory_mb"] = (pool->max_memory() - initial_memory) / (1024.0 * 1024.0);
}

BENCHMARK_REGISTER_F(FormatBenchFixture, ReadFullScan)
    ->ArgsProduct({
        {0, 1},      // Format: parquet, vortex
        {0, 1, 2}    // Memory: Low, Default, High
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(FormatBenchFixture, ReadProjection)
    ->ArgsProduct({
        {0, 1},        // Format: parquet, vortex
        {1, 2, 3, 4},  // Number of columns to read
        {0, 1, 2}      // Memory: Low, Default, High
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(FormatBenchFixture, ReadParallel)
    ->ArgsProduct({
        {0, 1},           // Format: parquet, vortex
        {1, 2, 4, 8, 0},  // Number of threads (0 = auto)
        {0, 1, 2}         // Memory: Low, Default, High
    })
    ->Unit(benchmark::kMillisecond);

// Random access (take) benchmark with memory configuration
BENCHMARK_DEFINE_F(FormatBenchFixture, ReadTake)(benchmark::State& st) {
  std::string format = formats[st.range(0)];
  size_t take_count = st.range(1);   // 10, 100, 1000, 10000
  int distribution = st.range(2);    // 0=sequential, 1=random, 2=clustered
  size_t memory_config = st.range(3);

  ConfigureMemory(properties_, memory_config);

  // Generate row indices based on distribution
  std::vector<int64_t> indices;
  switch (distribution) {
    case 0:
      indices = GenerateSequentialIndices(take_count, 0);
      break;
    case 1:
      indices = GenerateRandomIndices(take_count, total_rows);
      break;
    case 2:
      indices = GenerateClusteredIndices(take_count, total_rows);
      break;
  }

  auto* pool = arrow::default_memory_pool();
  int64_t initial_memory = pool->bytes_allocated();

  for (auto _ : st) {
    auto reader = Reader::create(cgs_, schema_, nullptr, properties_);
    auto table = reader->take(indices).ValueOrDie();
  }

  st.counters["peak_memory_mb"] = (pool->max_memory() - initial_memory) / (1024.0 * 1024.0);
}

BENCHMARK_REGISTER_F(FormatBenchFixture, ReadTake)
    ->ArgsProduct({
        {0, 1},                    // Format: parquet, vortex
        {10, 100, 1000, 10000},    // Number of rows to take
        {0, 1, 2},                 // Distribution: sequential, random, clustered
        {0, 1, 2}                  // Memory: Low, Default, High
    })
    ->Unit(benchmark::kMillisecond);
```

### 3. V2 vs V3 Reader Benchmark

**File:** `cpp/benchmark/benchmark_v2_v3.cpp`

**Purpose:** Compare performance between packed/ low-level read logic (v2) and top-level API (v3).

> **Note:** This benchmark is **Parquet-only** because `PackedRecordBatchReader` (v2) is a Parquet-specific low-level implementation. Vortex/Lance formats have their own low-level implementations and are not included in this comparison.

#### 3.1 Reader Layer Overview

| Version | Layer | Class | Description |
|---------|-------|-------|-------------|
| v2 | packed/ | `PackedRecordBatchReader` | Low-level implementation, directly operates Parquet files |
| v3 | Top-level | `Reader::get_record_batch_reader()` | High-level API, encapsulates column group coordination logic |
| v3 | Top-level | `Reader::get_chunk_reader()` | High-level API, chunk-based reading |

#### 3.2 Comparison Scenarios

| Scenario | v2 | v3 | Description |
|----------|-----|-----|-------------|
| RecordBatchReader | `PackedRecordBatchReader` | `Reader::get_record_batch_reader()` | Stream read all data |
| ChunkReader | N/A | `Reader::get_chunk_reader()` | Chunk-based random access |

#### 3.3 Test Configurations

| Config | Rows | Vector Dim | Files | Description |
|--------|------|------------|-------|-------------|
| Small | 10,000 | 128 | 1 | Single file, small dataset |
| Medium | 100,000 | 128 | 1 | Single file, medium dataset |
| Large | 1,000,000 | 128 | 1 | Single file, large dataset |
| MultiFile | 100,000 | 128 | 10 | Multi-file dataset |

#### 3.4 Metrics to Collect

- Time to read all data (ms)
- Read throughput (MB/s)
- Rows per second
- API overhead (v3_time - v2_time) / v2_time

#### 3.5 Benchmark Definition

```cpp
// v2: PackedRecordBatchReader (low-level)
BENCHMARK_DEFINE_F(V2V3BenchFixture, V2_PackedRecordBatchReader)(benchmark::State& st) {
  size_t config_idx = st.range(0);

  for (auto _ : st) {
    PackedRecordBatchReader reader(fs_, paths_, schema_, buffer_size_, reader_props_);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (reader.ReadNext(&batch).ok() && batch) {
      // consume batch
    }
  }
}

// v3: Reader::get_record_batch_reader() (top-level)
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_RecordBatchReader)(benchmark::State& st) {
  size_t config_idx = st.range(0);

  for (auto _ : st) {
    auto reader = Reader::create(cgs_, schema_, nullptr, properties_);
    auto batch_reader = reader->get_record_batch_reader().ValueOrDie();
    std::shared_ptr<arrow::RecordBatch> batch;
    while (batch_reader->ReadNext(&batch).ok() && batch) {
      // consume batch
    }
  }
}

// v3: Reader::get_chunk_reader() (top-level chunk access)
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_ChunkReader)(benchmark::State& st) {
  size_t config_idx = st.range(0);

  for (auto _ : st) {
    auto reader = Reader::create(cgs_, schema_, nullptr, properties_);
    auto chunk_reader = reader->get_chunk_reader(0).ValueOrDie();
    for (size_t i = 0; i < chunk_reader->total_number_of_chunks(); ++i) {
      auto batch = chunk_reader->get_chunk(i).ValueOrDie();
      // consume batch
    }
  }
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchReader)
    ->Args({0})   // Small
    ->Args({1})   // Medium
    ->Args({2})   // Large
    ->Args({3})   // MultiFile
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_RecordBatchReader)
    ->Args({0})   // Small
    ->Args({1})   // Medium
    ->Args({2})   // Large
    ->Args({3})   // MultiFile
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_ChunkReader)
    ->Args({0})   // Small
    ->Args({1})   // Medium
    ->Args({2})   // Large
    ->Args({3})   // MultiFile
    ->Unit(benchmark::kMillisecond);
```

#### 3.6 Expected Insights

This benchmark helps answer:
- How much performance overhead does the v3 top-level API have compared to v2 low-level?
- Which is more suitable for different scenarios: `get_record_batch_reader()` or `get_chunk_reader()`?
- Is there significant overhead from v3's column group coordination logic in multi-file scenarios?

### 4. Storage Layer Comparison Benchmark

**File:** `cpp/benchmark/benchmark_storage_layer.cpp`

**Purpose:** Compare Milvus-Storage (with Transaction layer) against Lance Native to evaluate the end-to-end storage system performance.

#### 4.1 Architecture Comparison

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│       Milvus-Storage                │         Lance Native                │
├─────────────────────────────────────┼─────────────────────────────────────┤
│  Write + Commit                     │  Write + Commit                     │
│  ─────────────────                  │  ─────────────────                  │
│  Writer::write(batch)               │  lance::write_dataset(uri, stream)  │
│         ↓                           │         ↓                           │
│  Writer::close()                    │    (built-in commit)                │
│         ↓                           │                                     │
│  Transaction::Commit()              │                                     │
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│  Open + Read                        │  Open + Read                        │
│  ────────────                       │  ────────────                       │
│  Transaction::Open(fs, path)        │  lance::open_dataset(uri)           │
│         ↓                           │         ↓                           │
│  Reader::get_record_batch_reader()  │  dataset.scan().to_table()          │
│                                     │                                     │
├─────────────────────────────────────┼─────────────────────────────────────┤
│  Random Access (Take)               │  Random Access (Take)               │
│  ────────────────────               │  ────────────────────               │
│  Transaction::Open(fs, path)        │  lance::open_dataset(uri)           │
│         ↓                           │         ↓                           │
│  Reader::take(indices)              │  dataset.take(indices)              │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

#### 4.2 Lance FFI Interface

The benchmark directly uses the lance bridge FFI layer (`cpp/src/format/bridge/rust/`) to call Lance Native APIs:

```cpp
#include "format/bridge/rust/include/lance_bridge.h"

using namespace milvus_storage::lance;

// Write: Export RecordBatches to ArrowArrayStream, then call WriteDataset
ArrowArrayStream stream;
ExportBatchesToArrowStream(batches, schema, &stream);
auto dataset = BlockingDataset::WriteDataset(uri, &stream);

// Read: Open dataset and scan (high-level API, no fragment handling needed)
auto dataset = BlockingDataset::Open(uri);
auto out_stream = dataset->Scan(batch_size);  // Returns ArrowArrayStream

// Take: Random access by indices (high-level API)
auto take_stream = dataset->Take(indices, batch_size);  // Returns ArrowArrayStream
```

> **Note:** If `BlockingDataset::Scan()` and `BlockingDataset::Take()` don't exist yet, they should be added to the lance bridge to simplify the benchmark. These high-level APIs internally handle fragment iteration.

#### 4.3 Test Configurations

| Config | Rows | Vector Dim | Batches | Description |
|--------|------|------------|---------|-------------|
| Small | 4,096 | 128 | 1 | Single batch write |
| Medium | 40,960 | 128 | 10 | Multi-batch write |
| Large | 409,600 | 128 | 100 | Large dataset |
| Append | 4,096 x 10 | 128 | 10 | Incremental appends |

#### 4.4 Comparison Scenarios

| Scenario | Milvus-Storage | Lance Native | Measures |
|----------|----------------|--------------|----------|
| **Write + Commit** | Writer + Transaction::Commit() | write_dataset() | Write throughput with transaction overhead |
| **Append + Commit** | Writer + Transaction::Commit() (multiple) | write_stream() (append) | Incremental write performance |
| **Open + Read** | Transaction::Open() + Reader::get_record_batch_reader() | open_dataset() + scan() | Read throughput with manifest loading |
| **Random Access** | Transaction::Open() + Reader::take() | open_dataset() + take() | Point query performance |

#### 4.5 Mixed Format Configurations

Test Milvus-Storage performance with mixed Parquet + Vortex storage. Mixed storage is achieved by writing different column groups with different formats, then combining them through Transaction's `AddColumnGroup()` / `AppendFiles()` operations.

**Mixed by Column Group:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Manifest                                     │
├─────────────────────────────────────────────────────────────────┤
│  Column Group 1 (Parquet)     │  Column Group 2 (Vortex)       │
│  ├── id (int64)               │  └── vector (float[128])       │
│  ├── name (string)            │                                │
│  └── value (double)           │                                │
└─────────────────────────────────────────────────────────────────┘

Write Flow:
1. Write scalar columns (id, name, value) with Parquet format
2. Write vector column with Vortex format
3. Use Transaction::AddColumnGroup() to combine them into one manifest
```

**Format Configuration:**

| Format Type | Value | Description |
|-------------|-------|-------------|
| PARQUET | 0 | Pure Parquet (all columns) |
| VORTEX | 1 | Pure Vortex (all columns) |
| MIXED | 2 | Mixed (scalar:Parquet, vector:Vortex) |

**Mixed Format Code Example:**

```cpp
// Format types for Storage Layer benchmark
enum FormatType {
  PARQUET = 0,
  VORTEX = 1,
  MIXED = 2      // Mixed by Column Group (scalar:parquet, vector:vortex)
};

// Write mixed format data
void WriteMixedFormat(const ArrowFileSystemPtr& fs,
                      const std::string& base_path,
                      const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                      std::shared_ptr<arrow::Schema> schema,
                      const Properties& properties) {
  // Step 1: Write scalar columns with Parquet
  auto scalar_schema = arrow::schema({
      schema->field(0),  // id
      schema->field(1),  // name
      schema->field(2)   // value
  });
  auto scalar_policy = CreateSinglePolicy(LOON_FORMAT_PARQUET, scalar_schema);
  auto scalar_writer = Writer::create(base_path, scalar_schema, std::move(scalar_policy), properties);
  for (const auto& batch : batches) {
    auto scalar_batch = ProjectColumns(batch, {0, 1, 2});
    scalar_writer->write(scalar_batch);
  }
  auto scalar_cgs = scalar_writer->close().ValueOrDie();  // std::shared_ptr<ColumnGroups>

  // Step 2: Write vector column with Vortex
  auto vector_schema = arrow::schema({schema->field(3)});  // vector
  auto vector_policy = CreateSinglePolicy(LOON_FORMAT_VORTEX, vector_schema);
  auto vector_writer = Writer::create(base_path, vector_schema, std::move(vector_policy), properties);
  for (const auto& batch : batches) {
    auto vector_batch = ProjectColumns(batch, {3});
    vector_writer->write(vector_batch);
  }
  auto vector_cgs = vector_writer->close().ValueOrDie();  // std::shared_ptr<ColumnGroups>

  // Step 3: Combine via Transaction
  // ColumnGroups is std::vector<std::shared_ptr<ColumnGroup>>
  auto txn = Transaction::Open(fs, base_path).ValueOrDie();
  for (const auto& cg : *scalar_cgs) {
    txn->AddColumnGroup(cg);   // Parquet column group
  }
  for (const auto& cg : *vector_cgs) {
    txn->AddColumnGroup(cg);   // Vortex column group
  }
  txn->Commit();
}

// Helper: Project specific columns from RecordBatch
std::shared_ptr<arrow::RecordBatch> ProjectColumns(
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::vector<int>& column_indices) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int idx : column_indices) {
    arrays.push_back(batch->column(idx));
    fields.push_back(batch->schema()->field(idx));
  }
  return arrow::RecordBatch::Make(arrow::schema(fields), batch->num_rows(), arrays);
}
```

#### 4.6 Benchmark Definition

```cpp
class StorageLayerFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& st) override {
    // Initialize properties from environment variables
    auto status = InitTestProperties(properties_);
    if (!status.ok()) {
      st.SkipWithError(status.ToString().c_str());
      return;
    }

    // Get filesystem (local or cloud based on STORAGE_TYPE)
    auto fs_result = GetFileSystem(properties_);
    if (!fs_result.ok()) {
      st.SkipWithError(fs_result.status().ToString().c_str());
      return;
    }
    fs_ = std::move(fs_result).ValueOrDie();

    // Initialize test data
    schema_ = CreateTestSchema();
    batches_ = GenerateTestBatches(num_rows_, batch_size_);

    // Setup paths
    milvus_storage_path_ = GetTestBasePath("milvus_storage_bench");
    lance_path_ = GetTestBasePath("lance_bench");
  }

  void TearDown(benchmark::State& st) override {
    // Clean up test directories
    DeleteTestDir(fs_, milvus_storage_path_);
    DeleteTestDir(fs_, lance_path_);
  }

 protected:
  ArrowFileSystemPtr fs_;
  api::Properties properties_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::string milvus_storage_path_;
  std::string lance_path_;
  size_t num_rows_ = 4096;
  size_t batch_size_ = 1024;
  size_t total_rows_ = 409600;  // For Take benchmark (Large config)
};

//=============================================================================
// Write + Commit Benchmarks
//=============================================================================

// Milvus-Storage: Writer + Transaction (supports mixed formats)
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_WriteCommit)(benchmark::State& st) {
  FormatType format_type = static_cast<FormatType>(st.range(0));  // 0=parquet, 1=vortex, 2=mixed
  size_t config_idx = st.range(1);

  for (auto _ : st) {
    if (format_type == MIXED) {
      // Mixed by Column Group: scalar columns (Parquet) + vector column (Vortex)
      // Step 1: Write scalar columns with Parquet
      auto scalar_schema = GetScalarSchema(schema_);
      auto scalar_policy = CreateSinglePolicy(LOON_FORMAT_PARQUET, scalar_schema);
      auto scalar_writer = Writer::create(milvus_storage_path_, scalar_schema,
                                          std::move(scalar_policy), properties_);
      for (const auto& batch : batches_) {
        scalar_writer->write(ProjectColumns(batch, {0, 1, 2}));  // id, name, value
      }
      auto scalar_cgs = scalar_writer->close().ValueOrDie();  // std::shared_ptr<ColumnGroups>

      // Step 2: Write vector column with Vortex
      auto vector_schema = GetVectorSchema(schema_);
      auto vector_policy = CreateSinglePolicy(LOON_FORMAT_VORTEX, vector_schema);
      auto vector_writer = Writer::create(milvus_storage_path_, vector_schema,
                                          std::move(vector_policy), properties_);
      for (const auto& batch : batches_) {
        vector_writer->write(ProjectColumns(batch, {3}));  // vector
      }
      auto vector_cgs = vector_writer->close().ValueOrDie();  // std::shared_ptr<ColumnGroups>

      // Step 3: Combine via Transaction
      // ColumnGroups is std::vector<std::shared_ptr<ColumnGroup>>
      auto txn = Transaction::Open(fs_, milvus_storage_path_).ValueOrDie();
      for (const auto& cg : *scalar_cgs) {
        txn->AddColumnGroup(cg);
      }
      for (const auto& cg : *vector_cgs) {
        txn->AddColumnGroup(cg);
      }
      txn->Commit();
    } else {
      // Pure format (PARQUET or VORTEX)
      std::string format = (format_type == PARQUET) ? LOON_FORMAT_PARQUET : LOON_FORMAT_VORTEX;
      auto policy = CreateSinglePolicy(format, schema_);
      auto writer = Writer::create(milvus_storage_path_, schema_, std::move(policy), properties_);
      for (const auto& batch : batches_) {
        writer->write(batch);
      }
      auto cgs = writer->close().ValueOrDie();  // std::shared_ptr<ColumnGroups>

      // ColumnGroups is std::vector<std::shared_ptr<ColumnGroup>>
      auto txn = Transaction::Open(fs_, milvus_storage_path_).ValueOrDie();
      for (const auto& cg : *cgs) {
        txn->AddColumnGroup(cg);
      }
      txn->Commit();
    }
  }
}

// Lance Native: WriteDataset (built-in transaction)
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_WriteCommit)(benchmark::State& st) {
  size_t config_idx = st.range(0);

  for (auto _ : st) {
    // Export batches to ArrowArrayStream
    ArrowArrayStream stream;
    ExportBatchesToArrowStream(batches_, schema_, &stream);

    // Write dataset (includes built-in commit)
    auto dataset = milvus_storage::lance::BlockingDataset::WriteDataset(
        lance_path_, &stream);
  }
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_WriteCommit)
    ->ArgsProduct({
        {0, 1, 2},     // FormatType: parquet, vortex, mixed
        {0, 1, 2, 3}   // Config: Small, Medium, Large, Append
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_WriteCommit)
    ->Args({0})   // Small
    ->Args({1})   // Medium
    ->Args({2})   // Large
    ->Args({3})   // Append
    ->Unit(benchmark::kMillisecond);

//=============================================================================
// Open + Read Benchmarks
//=============================================================================

// Milvus-Storage: Transaction + Reader (supports mixed formats)
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_OpenRead)(benchmark::State& st) {
  FormatType format_type = static_cast<FormatType>(st.range(0));  // 0=parquet, 1=vortex, 2=mixed
  size_t config_idx = st.range(1);

  // Pre-write test data with specified format configuration
  PrepareTestData(format_type, config_idx);

  for (auto _ : st) {
    // Open transaction (load manifest)
    // Manifest contains column group info with format metadata
    auto txn = Transaction::Open(fs_, milvus_storage_path_).ValueOrDie();
    auto manifest = txn->GetManifest().ValueOrDie();
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());

    // Reader automatically handles mixed formats based on manifest
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    auto batch_reader = reader->get_record_batch_reader().ValueOrDie();
    auto table = arrow::Table::FromRecordBatchReader(batch_reader.get()).ValueOrDie();
  }
}

// Lance Native: Open + Scan (high-level API)
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_OpenRead)(benchmark::State& st) {
  size_t config_idx = st.range(0);

  // Pre-write test data
  PrepareLanceData(config_idx);

  for (auto _ : st) {
    // Open dataset and scan all data
    auto dataset = milvus_storage::lance::BlockingDataset::Open(lance_path_);
    auto stream = dataset->Scan(batch_size_);

    // Consume stream to read all data
    ConsumeArrowStream(&stream);
  }
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_OpenRead)
    ->ArgsProduct({
        {0, 1, 2},     // FormatType: parquet, vortex, mixed
        {0, 1, 2}      // Config: Small, Medium, Large
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_OpenRead)
    ->Args({0})   // Small
    ->Args({1})   // Medium
    ->Args({2})   // Large
    ->Unit(benchmark::kMillisecond);

//=============================================================================
// Random Access (Take) Benchmarks
//=============================================================================

// Milvus-Storage: Transaction + Reader::take() (supports mixed formats)
BENCHMARK_DEFINE_F(StorageLayerFixture, MilvusStorage_Take)(benchmark::State& st) {
  FormatType format_type = static_cast<FormatType>(st.range(0));  // 0=parquet, 1=vortex, 2=mixed
  size_t take_count = st.range(1);

  PrepareTestData(format_type, 2);  // Large config
  auto indices = GenerateRandomIndices(take_count, total_rows_);

  for (auto _ : st) {
    auto txn = Transaction::Open(fs_, milvus_storage_path_).ValueOrDie();
    auto manifest = txn->GetManifest().ValueOrDie();
    auto cgs = std::make_shared<ColumnGroups>(manifest->columnGroups());
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    auto table = reader->take(indices);
  }
}

// Lance Native: Open + Take (high-level API)
BENCHMARK_DEFINE_F(StorageLayerFixture, LanceNative_Take)(benchmark::State& st) {
  size_t take_count = st.range(0);

  PrepareLanceData(2);  // Large config
  auto indices = GenerateRandomIndices(take_count, total_rows_);

  for (auto _ : st) {
    // Open dataset and take by indices
    auto dataset = milvus_storage::lance::BlockingDataset::Open(lance_path_);
    auto stream = dataset->Take(indices, batch_size_);

    // Consume stream
    ConsumeArrowStream(&stream);
  }
}

BENCHMARK_REGISTER_F(StorageLayerFixture, MilvusStorage_Take)
    ->ArgsProduct({
        {0, 1, 2},                 // FormatType: parquet, vortex, mixed
        {100, 1000, 10000}         // Take count
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StorageLayerFixture, LanceNative_Take)
    ->Args({100})
    ->Args({1000})
    ->Args({10000})
    ->Unit(benchmark::kMillisecond);
```

#### 4.7 Metrics to Collect

| Metric | Unit | Description |
|--------|------|-------------|
| Write + Commit Time | ms | Total time for write and transaction commit |
| Read + Open Time | ms | Total time for manifest loading and data reading |
| Throughput | MB/s | Data processed per second |
| Rows/s | rows/s | Rows processed per second |

#### 4.8 Expected Insights

This benchmark helps answer:
- How does Milvus-Storage (Parquet/Vortex) performance compare to Lance Native at the storage layer?
- How does mixed format (Parquet + Vortex) performance compare to pure format?
- Is there significant manifest loading overhead in Milvus-Storage?
- Which system is more efficient for random access (take) operations?

## Memory Control

Memory control is a cross-cutting concern that applies to all benchmarks. This section describes how to configure and monitor memory usage during benchmark execution.

### Memory Configuration

#### Buffer Size Settings

| Parameter | Property Key | Default | Description |
|-----------|--------------|---------|-------------|
| Writer Buffer Size | `PROPERTY_WRITER_BUFFER_SIZE` | 64 MB | Buffer size for writing data |
| Record Batch Max Rows | `PROPERTY_READER_RECORD_BATCH_MAX_ROWS` | 8192 | Max number of rows per record batch |
| Record Batch Max Size | `PROPERTY_READER_RECORD_BATCH_MAX_SIZE` | 64 MB | Max size of each record batch |

**Configuration Example:**

```cpp
api::Properties properties;
SetValue(properties, PROPERTY_WRITER_BUFFER_SIZE, "67108864");           // 64 MB
SetValue(properties, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, "8192");     // 8192 rows
SetValue(properties, PROPERTY_READER_RECORD_BATCH_MAX_SIZE, "67108864"); // 64 MB
```

#### Memory Pool Configuration

Arrow memory pool can be used to track memory usage:

```cpp
#include <arrow/memory_pool.h>

// Use default memory pool with tracking
arrow::MemoryPool* pool = arrow::default_memory_pool();

// Track memory before and after operations
int64_t bytes_before = pool->bytes_allocated();
// ... perform operations ...
int64_t peak_memory = pool->max_memory();
```

### Memory Benchmark Scenarios

Test performance under different memory constraints:

| Scenario | Buffer Size | Batch Size | Description |
|----------|-------------|------------|-------------|
| Low Memory | 16 MB | 1024 | Memory-constrained environment |
| Default | 128 MB | 16384 | Standard configuration |
| High Memory | 256 MB | 32768 | Memory-rich environment |
| Streaming | 8 MB | 512 | Minimal memory footprint |

### Memory Metrics to Collect

| Metric | Unit | Description |
|--------|------|-------------|
| Peak Memory Usage | MB | Maximum memory allocated during operation |
| Average Memory Usage | MB | Average memory usage over benchmark duration |
| Memory Allocation Count | count | Number of memory allocations |
| Memory Efficiency | MB/s per MB | Throughput relative to memory used |

### Benchmark Definition with Memory Control

```cpp
class MemoryControlFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& st) override {
    buffer_size_ = st.range(0);  // Buffer size in bytes
    batch_size_ = st.range(1);   // Rows per batch

    // Configure properties
    SetValue(properties_, PROPERTY_WRITER_BUFFER_SIZE, std::to_string(buffer_size_));
    SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, std::to_string(batch_size_));
    SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_SIZE, std::to_string(buffer_size_));

    // Setup memory tracking
    pool_ = arrow::default_memory_pool();
    initial_allocated_ = pool_->bytes_allocated();
  }

  void TearDown(benchmark::State& st) override {
    // Report memory metrics
    int64_t peak_allocated = pool_->max_memory() - initial_allocated_;
    st.counters["peak_memory_mb"] = benchmark::Counter(
        peak_allocated / (1024.0 * 1024.0),
        benchmark::Counter::kDefaults
    );
  }

 protected:
  size_t buffer_size_;
  size_t batch_size_;
  arrow::MemoryPool* pool_;
  int64_t initial_allocated_;
  api::Properties properties_;
};

// Read benchmark with memory control
BENCHMARK_DEFINE_F(MemoryControlFixture, ReadWithMemoryLimit)(benchmark::State& st) {
  std::string format = formats[st.range(2)];

  for (auto _ : st) {
    auto reader = Reader::create(cgs_, schema_, nullptr, properties_);
    auto batch_reader = reader->get_record_batch_reader().ValueOrDie();
    auto table = arrow::Table::FromRecordBatchReader(batch_reader.get()).ValueOrDie();
  }

  // Report throughput per MB of memory
  double throughput_mb_s = bytes_read / elapsed_time / (1024.0 * 1024.0);
  double memory_mb = (pool_->max_memory() - initial_allocated_) / (1024.0 * 1024.0);
  st.counters["memory_efficiency"] = throughput_mb_s / memory_mb;
}

BENCHMARK_REGISTER_F(MemoryControlFixture, ReadWithMemoryLimit)
    ->ArgsProduct({
        {16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024},  // Buffer size
        {1024, 8192, 32768},                                       // Batch size
        {0, 1}                                                     // Format
    })
    ->Unit(benchmark::kMillisecond);
```

### Memory Control for Different Benchmark Types

| Benchmark | Memory Focus | Key Parameters |
|-----------|--------------|----------------|
| Write Performance | Write buffer, batch accumulation | Write buffer size, batch size |
| Read Performance | Read buffer, row group caching | Read buffer size, prefetch |
| Column Projection | Column buffer allocation | Per-column buffer size |
| Multi-threading | Per-thread memory pool | Thread count × buffer size |
| Random Access (Take) | Index buffer, result buffer | Take count, buffer size |
| Storage Layer | Transaction memory, manifest cache | Total memory limit |

### Expected Output

```
MemoryControlFixture/ReadWithMemoryLimit/16777216/1024/0    xxx ms    xxx    # 16MB/1K/parquet
MemoryControlFixture/ReadWithMemoryLimit/16777216/1024/1    xxx ms    xxx    # 16MB/1K/vortex
MemoryControlFixture/ReadWithMemoryLimit/67108864/8192/0    xxx ms    xxx    # 64MB/8K/parquet
MemoryControlFixture/ReadWithMemoryLimit/67108864/8192/1    xxx ms    xxx    # 64MB/8K/vortex
MemoryControlFixture/ReadWithMemoryLimit/268435456/32768/0  xxx ms    xxx    # 256MB/32K/parquet
MemoryControlFixture/ReadWithMemoryLimit/268435456/32768/1  xxx ms    xxx    # 256MB/32K/vortex
```

## Implementation Plan

### Phase 1: Infrastructure

1. **Create Format Benchmark Fixture** (`benchmark_format_common.h`)
   - Base fixture class with format selection
   - Common setup/teardown logic
   - Format availability checking (via `GenerateFormatTestPValuesIn()`)
   - Metrics collection helpers

2. **Update CMakeLists.txt**
   - Add new benchmark source files
   - Conditional compilation for Vortex/Lance benchmarks

### Phase 2: Format Layer Benchmarks

1. **Write Performance** (`benchmark_format_write.cpp`)
   - Implement `FormatWriteComparison` benchmark
   - Multiple data size configurations
   - Per-format throughput metrics

2. **Read Performance** (`benchmark_format_read.cpp`)
   - Implement `ReadFullScan` benchmark for full table scan
   - Implement `ReadProjection` benchmark for column projection
   - Implement `ReadParallel` benchmark for multi-threading
   - Implement `ReadTake` benchmark for random access
   - Per-format throughput metrics
   - Projection efficiency and parallel efficiency analysis

3. **V2 vs V3 Reader** (`benchmark_v2_v3.cpp`)
   - Implement `V2_PackedRecordBatchReader` benchmark (Parquet only)
   - Implement `V3_RecordBatchReader` benchmark
   - Implement `V3_ChunkReader` benchmark
   - API overhead analysis

### Phase 3: Storage Layer Benchmarks

1. **Storage Layer Comparison** (`benchmark_storage_layer.cpp`)
   - Implement `MilvusStorage_WriteCommit` benchmark (Parquet/Vortex + Transaction)
   - Implement `LanceNative_WriteCommit` benchmark (direct FFI calls)
   - Implement `MilvusStorage_OpenRead` benchmark
   - Implement `LanceNative_OpenRead` benchmark
   - Implement `MilvusStorage_Take` benchmark
   - Implement `LanceNative_Take` benchmark
   - Transaction overhead analysis

## File Structure

```
cpp/benchmark/
├── benchmark_main.cpp              # Entry point (existing)
├── benchmark_wr.cpp                # Legacy write/read (existing)
├── benchmark_footer_size.cpp       # Footer analysis (existing)
├── benchmark_format_common.h       # NEW: Common fixture and helpers
├── benchmark_format_write.cpp      # NEW: Format Layer - Write comparison + File size/compression
├── benchmark_format_read.cpp       # NEW: Format Layer - Read comparison + Column projection + Multi-threading + Random access
├── benchmark_v2_v3.cpp             # NEW: Format Layer - V2 (packed/) vs V3 (top-level) reader comparison
├── benchmark_storage_layer.cpp     # NEW: Storage Layer - Milvus-Storage vs Lance Native comparison
└── CMakeLists.txt                  # Updated
```

## Cloud Storage Support

All benchmarks support running on cloud storage backends. Storage backend is configured via environment variables.

### Supported Cloud Providers

| Provider | CLOUD_PROVIDER | ADDRESS Example |
|----------|----------------|-----------------|
| AWS S3 | aws | s3.us-west-2.amazonaws.com |
| Google Cloud Storage | gcp | storage.googleapis.com |
| Azure Blob | azure | xxx.blob.core.windows.net |
| Aliyun OSS | aliyun | oss-cn-hangzhou.aliyuncs.com |
| Tencent COS | tencent | cos.ap-guangzhou.myqcloud.com |
| Huawei OBS | huawei | obs.cn-north-4.myhuaweicloud.com |

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `STORAGE_TYPE` | Storage type: `local` or `remote` | `remote` |
| `CLOUD_PROVIDER` | Cloud provider name | `aws` |
| `ADDRESS` | Cloud storage endpoint | `s3.us-west-2.amazonaws.com` |
| `BUCKET_NAME` | Bucket/container name | `my-benchmark-bucket` |
| `ACCESS_KEY` | Access key ID | `AKIAIOSFODNN7EXAMPLE` |
| `SECRET_KEY` | Secret access key | `wJalrXUtnFEMI/K7MDENG/...` |
| `REGION` | Cloud region | `us-west-2` |

### Benchmark Fixture Cloud Support

```cpp
class FormatBenchFixture : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& st) override {
    // Initialize properties from environment variables
    // Works for both local and cloud storage
    auto status = InitTestProperties(properties_);
    if (!status.ok()) {
      st.SkipWithError(status.ToString().c_str());
      return;
    }

    // Get filesystem (local or cloud based on STORAGE_TYPE)
    auto fs_result = GetFileSystem(properties_);
    if (!fs_result.ok()) {
      st.SkipWithError(fs_result.status().ToString().c_str());
      return;
    }
    fs_ = std::move(fs_result).ValueOrDie();

    // Create test directory
    test_path_ = GetTestBasePath("benchmark_format");
    CreateTestDir(fs_, test_path_);
  }

  void TearDown(benchmark::State& st) override {
    DeleteTestDir(fs_, test_path_);
  }

 protected:
  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string test_path_;
};
```

### Running Benchmarks on Cloud Storage

**Local Storage (default):**

```bash
./build/Release/benchmark --benchmark_filter="Format"
```

**AWS S3:**

```bash
export STORAGE_TYPE=remote
export CLOUD_PROVIDER=aws
export ADDRESS=s3.us-west-2.amazonaws.com
export BUCKET_NAME=my-benchmark-bucket
export ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
export SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export REGION=us-west-2

./build/Release/benchmark --benchmark_filter="Format"
```

**Google Cloud Storage:**

```bash
export STORAGE_TYPE=remote
export CLOUD_PROVIDER=gcp
export ADDRESS=storage.googleapis.com
export BUCKET_NAME=my-benchmark-bucket
export ACCESS_KEY=xxx
export SECRET_KEY=xxx

./build/Release/benchmark --benchmark_filter="Format"
```

**Aliyun OSS:**

```bash
export STORAGE_TYPE=remote
export CLOUD_PROVIDER=aliyun
export ADDRESS=oss-cn-hangzhou.aliyuncs.com
export BUCKET_NAME=my-benchmark-bucket
export ACCESS_KEY=xxx
export SECRET_KEY=xxx

./build/Release/benchmark --benchmark_filter="Format"
```

### Cloud-Specific Considerations

1. **Network Latency**: Cloud storage benchmarks will show higher latency than local storage. Results are still comparable across formats on the same cloud backend.

2. **Cost**: Running benchmarks on cloud storage incurs costs. Consider using smaller data sizes for cloud tests.

3. **Parallelism**: Cloud storage may benefit more from multi-threading due to network I/O being the bottleneck.

4. **Cleanup**: Benchmark fixtures automatically clean up test data after each run to avoid storage costs.

## Build Configuration

### CMakeLists.txt Updates

```cmake
find_package(benchmark REQUIRED)

# Core benchmark sources (always included)
set(BENCHMARK_SOURCES
  benchmark_main.cpp
  benchmark_wr.cpp
  benchmark_footer_size.cpp
  benchmark_format_write.cpp
  benchmark_format_read.cpp
  benchmark_v2_v3.cpp
)

# Storage Layer benchmark (requires Lance bridge)
if(BUILD_LANCE_BRIDGE)
  list(APPEND BENCHMARK_SOURCES benchmark_storage_layer.cpp)
endif()

# Include test helper
list(APPEND BENCHMARK_SOURCES "${PROJECT_SOURCE_DIR}/test/test_env.cpp")

add_executable(benchmark ${BENCHMARK_SOURCES})

target_link_libraries(benchmark
  milvus-storage
  benchmark::benchmark
)

target_include_directories(benchmark PUBLIC ${PROJECT_SOURCE_DIR}/test/include)
```

### Makefile Integration

```makefile
# ===== Format Layer Benchmarks =====

# Run format comparison benchmarks
benchmark-format:
	./build/Release/benchmark --benchmark_filter="Format"

# Run with Vortex enabled
benchmark-format-vortex:
	WITH_VORTEX=True make build
	./build/Release/benchmark --benchmark_filter="Format"

# ===== Storage Layer Benchmarks =====

# Run storage layer comparison (requires Lance)
benchmark-storage:
	WITH_LANCE=True make build
	./build/Release/benchmark --benchmark_filter="MilvusStorage_|LanceNative_"

# Run all benchmarks (Format + Storage Layer)
benchmark-all:
	WITH_VORTEX=True WITH_LANCE=True make build
	./build/Release/benchmark

# ===== Reports =====

# Generate JSON report
benchmark-format-report:
	./build/Release/benchmark --benchmark_filter="Format" \
		--benchmark_format=json \
		--benchmark_out=benchmark_format_results.json

benchmark-storage-report:
	./build/Release/benchmark --benchmark_filter="MilvusStorage_|LanceNative_" \
		--benchmark_format=json \
		--benchmark_out=benchmark_storage_results.json
```

## Typical Benchmarks

Typical benchmarks provide a quick validation set with representative parameters. Instead of running all parameter combinations, users can run only the typical cases to get a quick overview of performance characteristics.

### Running Typical Benchmarks

```bash
./build/Release/benchmark --benchmark_filter="Typical/"
```

### Typical Benchmark List

#### Storage Layer (`benchmark_storage_layer.cpp`)

| Benchmark Name | Original Benchmark | Parameters |
|----------------|-------------------|------------|
| `Typical/MilvusStorage_Write_Parquet` | MilvusStorage_WriteCommit | Parquet + Medium |
| `Typical/MilvusStorage_Write_Vortex` | MilvusStorage_WriteCommit | Vortex + Medium |
| `Typical/MilvusStorage_Read_Parquet` | MilvusStorage_OpenRead | Parquet + Medium |
| `Typical/MilvusStorage_Read_Vortex` | MilvusStorage_OpenRead | Vortex + Medium |
| `Typical/MilvusStorage_Take_Parquet` | MilvusStorage_Take | Parquet + 1000 rows |
| `Typical/MilvusStorage_Take_Vortex` | MilvusStorage_Take | Vortex + 1000 rows |
| `Typical/Lance_Write` | LanceNative_WriteCommit | Medium |
| `Typical/Lance_Read` | LanceNative_OpenRead | Medium |
| `Typical/Lance_Take` | LanceNative_Take | 1000 rows |

#### Format Layer - Write (`benchmark_format_write.cpp`)

| Benchmark Name | Original Benchmark | Parameters |
|----------------|-------------------|------------|
| `Typical/FormatWrite_Parquet` | WriteComparison | Parquet + Medium + Default |
| `Typical/FormatWrite_Vortex` | WriteComparison | Vortex + Medium + Default |
| `Typical/Compression_Parquet` | CompressionAnalysis | Parquet + Medium |
| `Typical/Compression_Vortex` | CompressionAnalysis | Vortex + Medium |

#### Format Layer - Read (`benchmark_format_read.cpp`)

| Benchmark Name | Original Benchmark | Parameters |
|----------------|-------------------|------------|
| `Typical/ReadFullScan_Parquet` | ReadFullScan | Parquet + Default |
| `Typical/ReadFullScan_Vortex` | ReadFullScan | Vortex + Default |
| `Typical/ReadProjection_Parquet` | ReadProjection | Parquet + 1 col + Default |
| `Typical/ReadProjection_Vortex` | ReadProjection | Vortex + 1 col + Default |
| `Typical/ReadParallel_Parquet` | ReadParallel | Parquet + 4 threads + Default |
| `Typical/ReadParallel_Vortex` | ReadParallel | Vortex + 4 threads + Default |
| `Typical/ReadTake_Parquet` | ReadTake | Parquet + 1000 rows + Random + Default |
| `Typical/ReadTake_Vortex` | ReadTake | Vortex + 1000 rows + Random + Default |

#### V2V3 Layer (`benchmark_v2_v3.cpp`)

| Benchmark Name | Original Benchmark | Parameters |
|----------------|-------------------|------------|
| `Typical/V2_Reader` | V2_PackedRecordBatchReader | Medium |
| `Typical/V3_Reader` | V3_RecordBatchReader | Medium |
| `Typical/V2_Writer` | V2_PackedRecordBatchWriter | Medium |
| `Typical/V3_Writer` | V3_Writer | Medium |

### Typical Benchmark Design Rationale

- **Medium data size**: Provides meaningful performance data without excessive runtime
- **Parquet + Vortex pairs**: Enables direct format comparison in each category
- **Default memory config**: Uses standard settings (128 MB buffer, 16384 batch size)
- **Random distribution for Take**: Simulates typical vector search result access patterns
- **4 threads for parallel tests**: Common multi-core configuration

## Usage Examples

### Build Benchmarks

```bash
cd cpp
BUILD_TYPE=Release USE_ASAN=False WITH_UT=False WITH_VORTEX=True WITH_LANCE=True make build
```

### Run Typical Benchmarks (Recommended for Quick Validation)

```bash
./build/Release/benchmark --benchmark_filter="Typical/"
```

### Run All Format Benchmarks

```bash
./build/Release/benchmark --benchmark_filter="Format"
```

### Run Specific Benchmark

```bash
# ===== Format Layer Benchmarks =====

# Write comparison (includes file size/compression metrics)
./build/Release/benchmark --benchmark_filter="Write"

# Read full scan only
./build/Release/benchmark --benchmark_filter="ReadFullScan"

# Read projection only
./build/Release/benchmark --benchmark_filter="ReadProjection"

# Read parallel (multi-threading) only
./build/Release/benchmark --benchmark_filter="ReadParallel"

# Read take (random access) only
./build/Release/benchmark --benchmark_filter="ReadTake"

# All read benchmarks (full scan + projection + parallel + take)
./build/Release/benchmark --benchmark_filter="Read"

# V2 vs V3 reader comparison
./build/Release/benchmark --benchmark_filter="V2_|V3_"

# ===== Storage Layer Benchmarks =====

# All Storage Layer benchmarks (requires BUILD_LANCE_BRIDGE)
./build/Release/benchmark --benchmark_filter="MilvusStorage_|LanceNative_"

# Write + Commit comparison
./build/Release/benchmark --benchmark_filter="WriteCommit"

# Open + Read comparison
./build/Release/benchmark --benchmark_filter="OpenRead"

# Take (random access) comparison
./build/Release/benchmark --benchmark_filter="MilvusStorage_Take|LanceNative_Take"
```

### Generate Report

```bash
./build/Release/benchmark \
    --benchmark_filter="Format" \
    --benchmark_format=json \
    --benchmark_out=format_benchmark_$(date +%Y%m%d).json
```

## Expected Output

```
Running ./build/Release/benchmark
Run on (8 X 2400 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 256 KiB (x8)
  L3 Unified 12288 KiB (x1)
---------------------------------------------------------------------------
Benchmark                                         Time           Iterations
---------------------------------------------------------------------------
# Write benchmarks
FormatBenchFixture/WriteComparison/0/0          xxx ms              xxx    # parquet/Small
FormatBenchFixture/WriteComparison/1/0          xxx ms              xxx    # vortex/Small
FormatBenchFixture/WriteComparison/0/1          xxx ms              xxx    # parquet/Medium
FormatBenchFixture/WriteComparison/1/1          xxx ms              xxx    # vortex/Medium

# Read full scan benchmarks (single thread)
FormatBenchFixture/ReadFullScan/0               xxx ms              xxx    # parquet
FormatBenchFixture/ReadFullScan/1               xxx ms              xxx    # vortex

# Read projection benchmarks (format/num_columns)
FormatBenchFixture/ReadProjection/0/1           xxx ms              xxx    # parquet/1col
FormatBenchFixture/ReadProjection/0/2           xxx ms              xxx    # parquet/2col
FormatBenchFixture/ReadProjection/0/3           xxx ms              xxx    # parquet/3col
FormatBenchFixture/ReadProjection/0/4           xxx ms              xxx    # parquet/4col
FormatBenchFixture/ReadProjection/1/1           xxx ms              xxx    # vortex/1col
FormatBenchFixture/ReadProjection/1/2           xxx ms              xxx    # vortex/2col
FormatBenchFixture/ReadProjection/1/3           xxx ms              xxx    # vortex/3col
FormatBenchFixture/ReadProjection/1/4           xxx ms              xxx    # vortex/4col

# Read parallel benchmarks (format/num_threads)
FormatBenchFixture/ReadParallel/0/1             xxx ms              xxx    # parquet/1thread (baseline)
FormatBenchFixture/ReadParallel/0/2             xxx ms              xxx    # parquet/2thread
FormatBenchFixture/ReadParallel/0/4             xxx ms              xxx    # parquet/4thread
FormatBenchFixture/ReadParallel/0/8             xxx ms              xxx    # parquet/8thread
FormatBenchFixture/ReadParallel/0/0             xxx ms              xxx    # parquet/auto
FormatBenchFixture/ReadParallel/1/1             xxx ms              xxx    # vortex/1thread (baseline)
FormatBenchFixture/ReadParallel/1/2             xxx ms              xxx    # vortex/2thread
FormatBenchFixture/ReadParallel/1/4             xxx ms              xxx    # vortex/4thread
FormatBenchFixture/ReadParallel/1/8             xxx ms              xxx    # vortex/8thread
FormatBenchFixture/ReadParallel/1/0             xxx ms              xxx    # vortex/auto

# Read take (random access) benchmarks (format/take_count/distribution)
# Distribution: 0=sequential, 1=random, 2=clustered
FormatBenchFixture/ReadTake/0/100/0             xxx ms              xxx    # parquet/100rows/sequential
FormatBenchFixture/ReadTake/0/100/1             xxx ms              xxx    # parquet/100rows/random
FormatBenchFixture/ReadTake/0/100/2             xxx ms              xxx    # parquet/100rows/clustered
FormatBenchFixture/ReadTake/0/1000/0            xxx ms              xxx    # parquet/1000rows/sequential
FormatBenchFixture/ReadTake/0/1000/1            xxx ms              xxx    # parquet/1000rows/random
FormatBenchFixture/ReadTake/0/1000/2            xxx ms              xxx    # parquet/1000rows/clustered
FormatBenchFixture/ReadTake/1/100/0             xxx ms              xxx    # vortex/100rows/sequential
FormatBenchFixture/ReadTake/1/100/1             xxx ms              xxx    # vortex/100rows/random
FormatBenchFixture/ReadTake/1/100/2             xxx ms              xxx    # vortex/100rows/clustered
FormatBenchFixture/ReadTake/1/1000/0            xxx ms              xxx    # vortex/1000rows/sequential
FormatBenchFixture/ReadTake/1/1000/1            xxx ms              xxx    # vortex/1000rows/random
FormatBenchFixture/ReadTake/1/1000/2            xxx ms              xxx    # vortex/1000rows/clustered

# V2 vs V3 reader benchmarks (config: Small/Medium/Large/MultiFile)
V2V3BenchFixture/V2_PackedRecordBatchReader/0   xxx ms              xxx    # v2/Small
V2V3BenchFixture/V2_PackedRecordBatchReader/1   xxx ms              xxx    # v2/Medium
V2V3BenchFixture/V2_PackedRecordBatchReader/2   xxx ms              xxx    # v2/Large
V2V3BenchFixture/V2_PackedRecordBatchReader/3   xxx ms              xxx    # v2/MultiFile
V2V3BenchFixture/V3_RecordBatchReader/0         xxx ms              xxx    # v3-rb/Small
V2V3BenchFixture/V3_RecordBatchReader/1         xxx ms              xxx    # v3-rb/Medium
V2V3BenchFixture/V3_RecordBatchReader/2         xxx ms              xxx    # v3-rb/Large
V2V3BenchFixture/V3_RecordBatchReader/3         xxx ms              xxx    # v3-rb/MultiFile
V2V3BenchFixture/V3_ChunkReader/0               xxx ms              xxx    # v3-chunk/Small
V2V3BenchFixture/V3_ChunkReader/1               xxx ms              xxx    # v3-chunk/Medium
V2V3BenchFixture/V3_ChunkReader/2               xxx ms              xxx    # v3-chunk/Large
V2V3BenchFixture/V3_ChunkReader/3               xxx ms              xxx    # v3-chunk/MultiFile

# ===== Storage Layer Benchmarks =====
# Format types: 0=parquet, 1=vortex, 2=mixed (scalar:parquet + vector:vortex)

# Write + Commit benchmarks (format_type/config for MilvusStorage)
StorageLayerFixture/MilvusStorage_WriteCommit/0/0     xxx ms        xxx    # parquet/Small
StorageLayerFixture/MilvusStorage_WriteCommit/0/1     xxx ms        xxx    # parquet/Medium
StorageLayerFixture/MilvusStorage_WriteCommit/0/2     xxx ms        xxx    # parquet/Large
StorageLayerFixture/MilvusStorage_WriteCommit/1/0     xxx ms        xxx    # vortex/Small
StorageLayerFixture/MilvusStorage_WriteCommit/1/1     xxx ms        xxx    # vortex/Medium
StorageLayerFixture/MilvusStorage_WriteCommit/1/2     xxx ms        xxx    # vortex/Large
StorageLayerFixture/MilvusStorage_WriteCommit/2/0     xxx ms        xxx    # mixed/Small
StorageLayerFixture/MilvusStorage_WriteCommit/2/1     xxx ms        xxx    # mixed/Medium
StorageLayerFixture/MilvusStorage_WriteCommit/2/2     xxx ms        xxx    # mixed/Large
StorageLayerFixture/LanceNative_WriteCommit/0         xxx ms        xxx    # lance/Small
StorageLayerFixture/LanceNative_WriteCommit/1         xxx ms        xxx    # lance/Medium
StorageLayerFixture/LanceNative_WriteCommit/2         xxx ms        xxx    # lance/Large

# Open + Read benchmarks (format_type/config for MilvusStorage)
StorageLayerFixture/MilvusStorage_OpenRead/0/0        xxx ms        xxx    # parquet/Small
StorageLayerFixture/MilvusStorage_OpenRead/0/1        xxx ms        xxx    # parquet/Medium
StorageLayerFixture/MilvusStorage_OpenRead/0/2        xxx ms        xxx    # parquet/Large
StorageLayerFixture/MilvusStorage_OpenRead/1/0        xxx ms        xxx    # vortex/Small
StorageLayerFixture/MilvusStorage_OpenRead/1/1        xxx ms        xxx    # vortex/Medium
StorageLayerFixture/MilvusStorage_OpenRead/1/2        xxx ms        xxx    # vortex/Large
StorageLayerFixture/MilvusStorage_OpenRead/2/0        xxx ms        xxx    # mixed/Small
StorageLayerFixture/MilvusStorage_OpenRead/2/1        xxx ms        xxx    # mixed/Medium
StorageLayerFixture/MilvusStorage_OpenRead/2/2        xxx ms        xxx    # mixed/Large
StorageLayerFixture/LanceNative_OpenRead/0            xxx ms        xxx    # lance/Small
StorageLayerFixture/LanceNative_OpenRead/1            xxx ms        xxx    # lance/Medium
StorageLayerFixture/LanceNative_OpenRead/2            xxx ms        xxx    # lance/Large

# Take benchmarks (format_type/take_count for MilvusStorage)
StorageLayerFixture/MilvusStorage_Take/0/100          xxx ms        xxx    # parquet/100rows
StorageLayerFixture/MilvusStorage_Take/0/1000         xxx ms        xxx    # parquet/1000rows
StorageLayerFixture/MilvusStorage_Take/0/10000        xxx ms        xxx    # parquet/10000rows
StorageLayerFixture/MilvusStorage_Take/1/100          xxx ms        xxx    # vortex/100rows
StorageLayerFixture/MilvusStorage_Take/1/1000         xxx ms        xxx    # vortex/1000rows
StorageLayerFixture/MilvusStorage_Take/1/10000        xxx ms        xxx    # vortex/10000rows
StorageLayerFixture/MilvusStorage_Take/2/100          xxx ms        xxx    # mixed/100rows
StorageLayerFixture/MilvusStorage_Take/2/1000         xxx ms        xxx    # mixed/1000rows
StorageLayerFixture/MilvusStorage_Take/2/10000        xxx ms        xxx    # mixed/10000rows
StorageLayerFixture/LanceNative_Take/100              xxx ms        xxx    # lance/100rows
StorageLayerFixture/LanceNative_Take/1000             xxx ms        xxx    # lance/1000rows
StorageLayerFixture/LanceNative_Take/10000            xxx ms        xxx    # lance/10000rows
```

## Conditional Compilation

For benchmarks that require optional formats:

```cpp
#include "milvus-storage/common/config.h"

std::vector<std::string> GetAvailableFormats() {
  std::vector<std::string> formats;
  formats.push_back(LOON_FORMAT_PARQUET);  // Always available

#ifdef BUILD_VORTEX_BRIDGE
  formats.push_back(LOON_FORMAT_VORTEX);
#endif

// Note: Lance is read-only, so only include in read benchmarks
#ifdef BUILD_LANCE_BRIDGE
  // formats.push_back(LOON_FORMAT_LANCE_TABLE);  // For read benchmarks only
#endif

  return formats;
}
```

## Metrics and Reporting

### Key Performance Indicators (KPIs)

| Metric | Unit | Description |
|--------|------|-------------|
| Write Throughput | MB/s | Data written per second |
| Read Throughput | MB/s | Data read per second |
| Compression Ratio | ratio | compressed_size / raw_size |
| Metadata Overhead | % | metadata_size / total_size * 100 |
| Projection Efficiency | ratio | full_scan_time / projection_time |
| Parallel Speedup | ratio | single_thread_time / n_thread_time |
| Parallel Efficiency | % | (speedup / n_threads) * 100 |
| Random Access Latency | us/row | take_time / num_rows * 1000 |
| API Overhead | % | (v3_time - v2_time) / v2_time * 100 |
| Peak Memory Usage | MB | Maximum memory allocated during operation |
| Memory Efficiency | MB/s per MB | Throughput relative to memory used |

### Benchmark Counters

```cpp
// Add custom counters to benchmark state
st.counters["throughput_mb_s"] = benchmark::Counter(
    bytes_processed / (1024.0 * 1024.0),
    benchmark::Counter::kIsRate
);

st.counters["rows_per_sec"] = benchmark::Counter(
    total_rows,
    benchmark::Counter::kIsRate
);

st.counters["compression_ratio"] = benchmark::Counter(
    static_cast<double>(compressed_size) / raw_size,
    benchmark::Counter::kDefaults
);

// Multi-threading counters
st.counters["threads"] = benchmark::Counter(
    num_threads,
    benchmark::Counter::kDefaults
);

st.counters["speedup"] = benchmark::Counter(
    baseline_time / actual_time,
    benchmark::Counter::kDefaults
);

st.counters["parallel_efficiency"] = benchmark::Counter(
    (baseline_time / actual_time / num_threads) * 100.0,
    benchmark::Counter::kDefaults
);

// Memory counters
st.counters["peak_memory_mb"] = benchmark::Counter(
    peak_allocated / (1024.0 * 1024.0),
    benchmark::Counter::kDefaults
);

st.counters["memory_efficiency"] = benchmark::Counter(
    throughput_mb_s / memory_mb,
    benchmark::Counter::kDefaults
);
```

## Dependencies

- Google Benchmark 1.7.0 (existing dependency)
- Test environment helpers (`test_env.h`, `test_env.cpp`)
- Format bridge libraries:
  - Vortex bridge (optional, for Format Layer benchmarks)
  - Lance bridge (required for Storage Layer benchmarks, `BUILD_LANCE_BRIDGE`)
- Lance FFI interface (`cpp/src/format/bridge/rust/include/lance_bridge.h`)

## Testing Checklist

- [ ] Phase 1: Infrastructure
  - [ ] Create `benchmark_format_common.h` with common fixture
  - [ ] Update CMakeLists.txt for new files
  - [ ] Verify build with/without Vortex flag

- [ ] Phase 2: Format Layer Benchmarks
  - [ ] Implement write comparison benchmark (includes file size/compression metrics)
  - [ ] Implement read full scan benchmark
  - [ ] Implement read projection benchmark (column projection efficiency)
  - [ ] Implement read parallel benchmark (multi-threading scalability)
  - [ ] Implement read take benchmark (random access performance with different distributions)
  - [ ] Implement v2 vs v3 reader benchmark (PackedRecordBatchReader vs Reader API)
  - [ ] Verify all Format Layer benchmarks run correctly

- [ ] Phase 3: Storage Layer Benchmarks
  - [ ] Extend Lance bridge with high-level APIs if needed:
    - [ ] `BlockingDataset::Scan(batch_size)` - scan all data without fragment handling
    - [ ] `BlockingDataset::Take(indices, batch_size)` - random access without fragment handling
  - [ ] Implement `MilvusStorage_WriteCommit` benchmark (Parquet/Vortex/Mixed + Transaction)
  - [ ] Implement `LanceNative_WriteCommit` benchmark (direct FFI calls to lance bridge)
  - [ ] Implement `MilvusStorage_OpenRead` benchmark
  - [ ] Implement `LanceNative_OpenRead` benchmark
  - [ ] Implement `MilvusStorage_Take` benchmark
  - [ ] Implement `LanceNative_Take` benchmark
  - [ ] Implement Mixed Format support (scalar:Parquet + vector:Vortex via Transaction::AddColumnGroup)
  - [ ] Verify all Storage Layer benchmarks run correctly with BUILD_LANCE_BRIDGE

- [ ] Memory Control
  - [ ] Implement memory tracking in benchmark fixtures
  - [ ] Add buffer size and batch size parameters to benchmarks
  - [ ] Implement `ReadWithMemoryLimit` benchmark
  - [ ] Verify memory metrics collection (peak memory, allocation count)
  - [ ] Test different memory constraint scenarios

- [ ] Documentation
  - [ ] Update CLAUDE.md with benchmark commands
  - [ ] Add benchmark results interpretation guide

## References

- [Google Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Apache Parquet Format](https://parquet.apache.org/docs/)
- [Vortex Format](https://github.com/spiraldb/vortex)
- [Lance Format](https://github.com/lancedb/lance)
- Existing benchmark code: `cpp/benchmark/benchmark_wr.cpp`
- Lance bridge implementation: `cpp/src/format/bridge/rust/`
