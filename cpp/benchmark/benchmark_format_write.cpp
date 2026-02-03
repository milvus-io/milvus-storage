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

#include <arrow/filesystem/filesystem.h>

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;

//=============================================================================
// Write Performance Benchmark
//=============================================================================

class FormatWriteBenchmark : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);

    // Get schema from data loader
    schema_ = GetLoaderSchema();

    // Pre-load all batches for write benchmarks (measure pure write performance)
    auto reader_result = GetLoaderBatchReader();
    if (!reader_result.ok()) {
      st.SkipWithError(("Failed to get batch reader: " + reader_result.status().ToString()).c_str());
      return;
    }
    auto batch_reader = *reader_result;

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      auto status = batch_reader->ReadNext(&batch);
      if (!status.ok()) {
        st.SkipWithError(("Failed to read batch: " + status.ToString()).c_str());
        return;
      }
      if (!batch)
        break;
      batches_.push_back(batch);
      total_bytes_ += CalculateRawDataSize(batch);
      total_rows_ += batch->num_rows();
    }
  }

  void TearDown(::benchmark::State& st) override {
    // Clear pre-loaded batches to release memory
    batches_.clear();
    batches_.shrink_to_fit();
    schema_.reset();
    FormatBenchFixtureBase<>::TearDown(st);
  }

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  int64_t total_bytes_ = 0;
  int64_t total_rows_ = 0;
};

// Write comparison benchmark across formats
// Args: [format_idx, data_config_idx, memory_config_idx]
BENCHMARK_DEFINE_F(FormatWriteBenchmark, WriteComparison)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  // data_config_idx is ignored - we use data from loader
  size_t memory_config_idx = static_cast<size_t>(st.range(2));

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  MemoryConfig memory_config = MemoryConfig::FromIndex(memory_config_idx);

  // Configure memory settings
  ConfigureMemory(memory_config);

  // Track total bytes and rows for throughput calculation
  int64_t total_bytes_written = 0;
  int64_t total_rows_written = 0;

  for (auto _ : st) {
    std::string path = GetUniquePath(format);

    // Create policy using schema-based patterns from loader
    std::string patterns = GetSchemaBasePatterns();
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_), st);

    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      BENCH_ASSERT_STATUS_OK(writer->write(batch), st);
    }

    // Close and get column groups
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    total_bytes_written += total_bytes_;
    total_rows_written += total_rows_;
  }

  // Report metrics
  ReportThroughput(st, total_bytes_written, total_rows_written);

  // Add labels for better output readability
  st.SetLabel(format + "/" + GetDataDescription());
}

// Register write benchmark with all combinations
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->ArgsProduct({
        {0, 1},     // Format: parquet(0), vortex(1)
        {0, 1, 2},  // DataConfig: Small(0), Medium(1), Large(2)
        {1}         // MemoryConfig: Default(1) only for basic tests
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Extended write benchmark with all memory configurations (for detailed analysis)
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteComparisonExtended")
    ->ArgsProduct({
        {0, 1},    // Format: parquet(0), vortex(1)
        {0},       // DataConfig: Small(0) only
        {0, 1, 2}  // MemoryConfig: Low(0), Default(1), High(2)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// High-dimensional vector write benchmark
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteHighDim")
    ->ArgsProduct({
        {0, 1},  // Format: parquet(0), vortex(1)
        {3},     // DataConfig: HighDim(3)
        {1}      // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

// Long string write benchmark
BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("FormatWriteBenchmark/WriteLongString")
    ->ArgsProduct({
        {0, 1},  // Format: parquet(0), vortex(1)
        {4},     // DataConfig: LongString(4)
        {1}      // MemoryConfig: Default(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

//=============================================================================
// File Size / Compression Analysis Benchmark
//=============================================================================

// Helper function to calculate total file size from column groups
static arrow::Result<int64_t> CalculateFileSize(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                const std::shared_ptr<api::ColumnGroups>& cgs) {
  int64_t total_size = 0;
  for (const auto& cg : *cgs) {
    for (const auto& file : cg->files) {
      ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(file.path));
      total_size += file_info.size();
    }
  }
  return total_size;
}

// This benchmark measures file size and compression ratio
// The compression ratio is calculated by comparing file size against raw data size from loader
// Args: [format_idx, data_config_idx]
BENCHMARK_DEFINE_F(FormatWriteBenchmark, CompressionAnalysis)(::benchmark::State& st) {
  size_t format_idx = static_cast<size_t>(st.range(0));
  // data_config_idx is ignored - we use data from loader

  std::string format = GetFormatByIndex(format_idx);
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  std::string patterns = GetSchemaBasePatterns();

  // Baseline is raw data size from loader (calculated in SetUp via CalculateRawDataSize)
  int64_t baseline_size = total_bytes_;

  // Write files with the target format and measure
  int64_t total_file_size = 0;
  int64_t iteration_count = 0;

  for (auto _ : st) {
    std::string path = GetUniquePath(format);

    // Create policy and writer with target format (uses default zstd compression for parquet)
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, format, schema_), st);
    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      BENCH_ASSERT_STATUS_OK(writer->write(batch), st);
    }

    // Close and get column groups
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    // Calculate file size
    BENCH_ASSERT_AND_ASSIGN(auto file_size, CalculateFileSize(fs_, cgs), st);

    total_file_size += file_size;
    iteration_count++;
  }

  // Report compression metrics
  if (iteration_count > 0) {
    int64_t avg_file_size = total_file_size / iteration_count;

    // Compression ratio = file_size / raw_data_size
    // Values < 1.0 indicate good compression
    ReportCompressionRatio(st, baseline_size, avg_file_size);

    ReportSize(st, "file_size", avg_file_size);
    ReportSize(st, "baseline_size", baseline_size);
  }

  st.SetLabel(format + "/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->ArgsProduct({
        {0, 1},          // Format: parquet(0), vortex(1)
        {0, 1, 2, 3, 4}  // All DataConfigs
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);  // Few iterations since we're measuring file size

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("Typical/FormatWrite_Parquet")
    ->Args({0, 1, 1})  // Parquet + Medium + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatWriteBenchmark, WriteComparison)
    ->Name("Typical/FormatWrite_Vortex")
    ->Args({1, 1, 1})  // Vortex + Medium + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->Name("Typical/Compression_Parquet")
    ->Args({0, 1})  // Parquet + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

BENCHMARK_REGISTER_F(FormatWriteBenchmark, CompressionAnalysis)
    ->Name("Typical/Compression_Vortex")
    ->Args({1, 1})  // Vortex + Medium
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(3);

}  // namespace benchmark
}  // namespace milvus_storage
