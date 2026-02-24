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

#include "benchmark_format_common.h"

#include <thread>
#include <arrow/table.h>
#include "milvus-storage/thread_pool.h"

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;

//=============================================================================
// Read Performance Benchmark Base
//=============================================================================

class FormatReadBenchmark : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);

    // Get schema from data loader
    schema_ = GetLoaderSchema();
  }

  void TearDown(::benchmark::State& st) override {
    // Clear schema to release memory
    schema_.reset();
    // Release thread pool before base teardown
    ThreadPoolHolder::Release();
    FormatBenchFixtureBase<>::TearDown(st);
  }

  protected:
  // Prepare test data by writing to storage using streaming reader (memory-efficient)
  arrow::Status PrepareTestData(const std::string& format,
                                std::shared_ptr<ColumnGroups>& out_cgs,
                                std::string& out_path) {
    out_path = GetUniquePath(format + "_read_test");

    // Use schema-based policy to preserve column groups
    std::string patterns = GetSchemaBasePatterns();
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSchemaBasePolicy(patterns, format, schema_));

    auto writer = Writer::create(out_path, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create writer");
    }

    // Write using streaming reader (memory-efficient for large datasets)
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, GetLoaderBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch)
        break;
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }

    ARROW_ASSIGN_OR_RAISE(out_cgs, writer->close());

    return arrow::Status::OK();
  }

  // Get projection columns based on count (from schema)
  std::shared_ptr<std::vector<std::string>> GetProjection(size_t num_columns) {
    auto projection = std::make_shared<std::vector<std::string>>();
    for (size_t i = 0; i < std::min(num_columns, static_cast<size_t>(schema_->num_fields())); ++i) {
      projection->push_back(schema_->field(i)->name());
    }
    return projection;
  }

  std::shared_ptr<arrow::Schema> schema_;
};

//=============================================================================
// Full Scan Benchmark
//=============================================================================

// Full scan benchmark - read all rows and all columns
// Args: [format_idx, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadFullScan)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  size_t num_threads = static_cast<size_t>(st.range(1));
  size_t memory_config_idx = static_cast<size_t>(st.range(2));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  MemoryConfig memory_config = MemoryConfig::FromIndex(memory_config_idx);

  // Configure memory and thread pool
  ConfigureMemory(memory_config);
  ThreadPoolHolder::WithSingleton(static_cast<int>(num_threads));

  // Prepare test data using data loader
  std::shared_ptr<ColumnGroups> cgs;
  std::string path;
  BENCH_ASSERT_STATUS_OK(PrepareTestData(format, cgs, path), st);

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader(), st);

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader->ReadNext(&batch), st);
      if (batch == nullptr) {
        break;
      }
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(format + "/" + std::to_string(num_threads) + "T/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadFullScan)
    ->ArgsProduct({
        {0, 1},         // Format: parquet(0), vortex(1)
        {1, 4, 8, 16},  // Threads: 1, 4, 8, 16
        {1}             // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Column Projection Benchmark
//=============================================================================

// Column projection benchmark - read subset of columns
// Args: [format_idx, num_columns, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadProjection)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  size_t num_columns = static_cast<size_t>(st.range(1));
  size_t num_threads = static_cast<size_t>(st.range(2));
  size_t memory_config_idx = static_cast<size_t>(st.range(3));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  MemoryConfig memory_config = MemoryConfig::FromIndex(memory_config_idx);

  ConfigureMemory(memory_config);
  ThreadPoolHolder::WithSingleton(static_cast<int>(num_threads));

  // Prepare test data (write all columns)
  std::shared_ptr<ColumnGroups> cgs;
  std::string path;
  BENCH_ASSERT_STATUS_OK(PrepareTestData(format, cgs, path), st);

  // Get projection for specified number of columns
  auto projection = GetProjection(num_columns);

  // Create projected schema
  std::vector<std::shared_ptr<arrow::Field>> projected_fields;
  for (const auto& col_name : *projection) {
    auto field = schema_->GetFieldByName(col_name);
    if (field) {
      projected_fields.push_back(field);
    }
  }
  auto projected_schema = arrow::schema(projected_fields);

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, projected_schema, projection, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader(), st);

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader->ReadNext(&batch), st);
      if (batch == nullptr) {
        break;
      }
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);
  st.SetLabel(format + "/" + std::to_string(num_columns) + "cols/" + std::to_string(num_threads) + "T/" +
              GetDataDescription());
}

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadProjection)
    ->ArgsProduct({
        {0, 1},        // Format: parquet(0), vortex(1)
        {1, 2, 3, 4},  // Number of columns: 1, 2, 3, 4
        {1, 8},        // Threads: 1, 8
        {1}            // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Random Access (Take) Benchmark
//=============================================================================

// Random access benchmark - read specific rows by indices
// Args: [format_idx, take_count, distribution, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadTake)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  size_t take_count = static_cast<size_t>(st.range(1));
  int distribution = static_cast<int>(st.range(2));
  size_t num_threads = static_cast<size_t>(st.range(3));
  size_t memory_config_idx = static_cast<size_t>(st.range(4));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  MemoryConfig memory_config = MemoryConfig::FromIndex(memory_config_idx);

  ConfigureMemory(memory_config);
  ThreadPoolHolder::WithSingleton(static_cast<int>(num_threads));

  // Prepare test data
  std::shared_ptr<ColumnGroups> cgs;
  std::string path;
  BENCH_ASSERT_STATUS_OK(PrepareTestData(format, cgs, path), st);

  // Generate indices based on distribution
  IndexDistribution dist = static_cast<IndexDistribution>(distribution);
  auto indices = GenerateIndices(dist, take_count, GetLoaderNumRows());

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto table, reader->take(indices), st);

    total_rows_read += table->num_rows();
    // Estimate bytes read
    arrow::TableBatchReader batch_reader(*table);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader.ReadNext(&batch), st);
      if (batch == nullptr) {
        break;
      }
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.counters["rows_taken"] = ::benchmark::Counter(static_cast<double>(take_count), ::benchmark::Counter::kDefaults);
  st.counters["threads"] = ::benchmark::Counter(static_cast<double>(num_threads), ::benchmark::Counter::kDefaults);

  st.SetLabel(format + "/" + std::to_string(take_count) + "rows/" + IndexDistributionName(dist) + "/" +
              std::to_string(num_threads) + "T/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadTake)
    ->ArgsProduct({
        {0, 1},                  // Format: parquet(0), vortex(1)
        {10, 100, 1000, 10000},  // Take count
        {0, 1, 2},               // Distribution: sequential(0), random(1), clustered(2)
        {1, 8},                  // Threads: 1, 8
        {1}                      // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadFullScan)
    ->Name("Typical/ReadFullScan_Parquet")
    ->Args({0, 8, 1})  // Parquet + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadFullScan)
    ->Name("Typical/ReadFullScan_Vortex")
    ->Args({1, 8, 1})  // Vortex + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadProjection)
    ->Name("Typical/ReadProjection_Parquet")
    ->Args({0, 1, 8, 1})  // Parquet + 1 col + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadProjection)
    ->Name("Typical/ReadProjection_Vortex")
    ->Args({1, 1, 8, 1})  // Vortex + 1 col + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadTake)
    ->Name("Typical/ReadTake_Parquet")
    ->Args({0, 1000, 1, 8, 1})  // Parquet + 1000 rows + Random + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadTake)
    ->Name("Typical/ReadTake_Vortex")
    ->Args({1, 1000, 1, 8, 1})  // Vortex + 1000 rows + Random + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
