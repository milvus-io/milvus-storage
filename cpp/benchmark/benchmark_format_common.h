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

#pragma once

#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <random>
#include <algorithm>
#include <set>
#include <thread>
#include <atomic>
#include <chrono>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#endif

#include <arrow/filesystem/filesystem.h>
#include <arrow/api.h>
#include <arrow/memory_pool.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
#include "test_env.h"
#include "benchmark_data_loader.h"

namespace milvus_storage {
namespace benchmark {

//=============================================================================
// Benchmark Assertion Macros
//=============================================================================

#define BENCH_ASSERT_STATUS_OK(status, st)             \
  do {                                                 \
    if (!(status).ok()) {                              \
      (st).SkipWithError((status).ToString().c_str()); \
      return;                                          \
    }                                                  \
  } while (false)

#define BENCH_ASSERT_AND_ASSIGN_IMPL(status_name, lhs, rexpr, st) \
  auto status_name = (rexpr);                                     \
  BENCH_ASSERT_STATUS_OK(status_name.status(), st);               \
  lhs = std::move(status_name).ValueOrDie();

#define BENCH_ASSERT_AND_ASSIGN(lhs, rexpr, st) \
  BENCH_ASSERT_AND_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr, st)

#define BENCH_ASSERT_NOT_NULL(ptr, st)               \
  do {                                               \
    if ((ptr) == nullptr) {                          \
      (st).SkipWithError("Unexpected null pointer"); \
      return;                                        \
    }                                                \
  } while (false)

//=============================================================================
// Benchmark Metrics Helpers
//=============================================================================

// Helper struct for formatted size with unit
struct FormattedSize {
  double value;
  std::string unit;
};

// Format bytes to appropriate unit (B, KB, MB, GB, TB)
inline FormattedSize FormatBytes(int64_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_idx = 0;
  double value = static_cast<double>(bytes);

  while (value >= 1024.0 && unit_idx < 4) {
    value /= 1024.0;
    unit_idx++;
  }

  return {value, units[unit_idx]};
}

// Report size with automatic unit selection
// Key will be "{name}({unit})", e.g., "file_size(MB)"
inline void ReportSize(::benchmark::State& st,
                       const std::string& name,
                       int64_t bytes,
                       ::benchmark::Counter::Flags flags = ::benchmark::Counter::kAvgIterations) {
  auto formatted = FormatBytes(bytes);
  st.counters[name + "(" + formatted.unit + ")"] = ::benchmark::Counter(formatted.value, flags);
}

inline void ReportThroughput(::benchmark::State& st, int64_t bytes_processed, int64_t rows_processed) {
  auto formatted = FormatBytes(bytes_processed);
  st.counters["throughput(" + formatted.unit + "/s)"] =
      ::benchmark::Counter(formatted.value, ::benchmark::Counter::kIsRate);
  st.counters["rows/s"] = ::benchmark::Counter(static_cast<double>(rows_processed), ::benchmark::Counter::kIsRate);
}

inline void ReportCompressionRatio(::benchmark::State& st, int64_t raw_size, int64_t compressed_size) {
  if (raw_size > 0) {
    st.counters["compression_ratio"] = ::benchmark::Counter(
        static_cast<double>(compressed_size) / static_cast<double>(raw_size), ::benchmark::Counter::kDefaults);
  }
}

//=============================================================================
// Data Configuration Structures
//=============================================================================

// Data size configurations as per design doc
struct DataSizeConfig {
  std::string name;
  size_t num_rows;
  size_t vector_dim;
  size_t string_length;

  static DataSizeConfig Small() { return {"Small", 4096, 128, 128}; }
  static DataSizeConfig Medium() { return {"Medium", 40960, 128, 128}; }
  static DataSizeConfig Large() { return {"Large", 409600, 128, 128}; }
  static DataSizeConfig HighDim() { return {"HighDim", 4096, 768, 128}; }
  static DataSizeConfig LongString() { return {"LongString", 4096, 128, 1024}; }

  static std::vector<DataSizeConfig> All() { return {Small(), Medium(), Large(), HighDim(), LongString()}; }

  static DataSizeConfig FromIndex(size_t idx) {
    auto all = All();
    return idx < all.size() ? all[idx] : Small();
  }
};

// Memory configurations as per design doc
struct MemoryConfig {
  std::string name;
  size_t buffer_size;
  size_t batch_size;

  static MemoryConfig Low() { return {"Low", 16ULL * 1024 * 1024, 1024}; }
  static MemoryConfig Default() { return {"Default", 128ULL * 1024 * 1024, 16384}; }
  static MemoryConfig High() { return {"High", 256ULL * 1024 * 1024, 32768}; }

  static std::vector<MemoryConfig> All() { return {Low(), Default(), High()}; }

  static MemoryConfig FromIndex(size_t idx) {
    auto all = All();
    return idx < all.size() ? all[idx] : Default();
  }
};

//=============================================================================
// Index Distribution Generators for Take Benchmarks
//=============================================================================

// Generate sequential indices starting from a given start position
inline std::vector<int64_t> GenerateSequentialIndices(size_t count, int64_t start = 0) {
  std::vector<int64_t> indices(count);
  std::iota(indices.begin(), indices.end(), start);
  return indices;
}

// Generate uniformly distributed random indices
inline std::vector<int64_t> GenerateRandomIndices(size_t count, int64_t max_value, uint32_t seed = 42) {
  std::vector<int64_t> indices;
  std::set<int64_t> seen;
  std::mt19937 gen(seed);
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

// Generate clustered indices (multiple small contiguous clusters)
inline std::vector<int64_t> GenerateClusteredIndices(size_t count,
                                                     int64_t max_value,
                                                     size_t cluster_size = 5,
                                                     uint32_t seed = 42) {
  std::vector<int64_t> indices;
  std::set<int64_t> seen;
  size_t num_clusters = (count + cluster_size - 1) / cluster_size;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int64_t> dist(0, max_value - static_cast<int64_t>(cluster_size));

  while (indices.size() < count) {
    int64_t start = dist(gen);
    for (size_t j = 0; j < cluster_size && indices.size() < count; ++j) {
      int64_t idx = start + static_cast<int64_t>(j);
      if (idx < max_value && seen.insert(idx).second) {
        indices.push_back(idx);
      }
    }
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

// Distribution type enum
enum class IndexDistribution { Sequential = 0, Random = 1, Clustered = 2 };

inline std::vector<int64_t> GenerateIndices(IndexDistribution dist,
                                            size_t count,
                                            int64_t max_value,
                                            uint32_t seed = 42) {
  switch (dist) {
    case IndexDistribution::Sequential:
      return GenerateSequentialIndices(count, 0);
    case IndexDistribution::Random:
      return GenerateRandomIndices(count, max_value, seed);
    case IndexDistribution::Clustered:
      return GenerateClusteredIndices(count, max_value, 5, seed);
    default:
      return GenerateRandomIndices(count, max_value, seed);
  }
}

inline const char* IndexDistributionName(IndexDistribution dist) {
  switch (dist) {
    case IndexDistribution::Sequential:
      return "Sequential";
    case IndexDistribution::Random:
      return "Random";
    case IndexDistribution::Clustered:
      return "Clustered";
    default:
      return "Unknown";
  }
}

//=============================================================================
// Format Utilities
//=============================================================================

// Get list of available formats based on build configuration
inline std::vector<std::string> GetAvailableFormats() { return {LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX}; }

// Get format name by index
inline std::string GetFormatByIndex(size_t idx) {
  auto formats = GetAvailableFormats();
  assert(idx < formats.size() && "Format index out of range");
  return formats[idx];
}

// Check if a format is available
inline bool IsFormatAvailable(const std::string& format) {
  auto formats = GetAvailableFormats();
  return std::find(formats.begin(), formats.end(), format) != formats.end();
}

//=============================================================================
// Memory Tracking Utilities (System-level)
//=============================================================================

// Get max RSS (Resident Set Size) in bytes using getrusage
// Note: ru_maxrss is the maximum RSS during the process lifetime
inline int64_t GetMaxRSS() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
    // macOS: ru_maxrss is in bytes
    return static_cast<int64_t>(usage.ru_maxrss);
#else
    // Linux: ru_maxrss is in KB
    return static_cast<int64_t>(usage.ru_maxrss) * 1024;
#endif
  }
  return 0;
}

class MemoryTracker {
  public:
  MemoryTracker() = default;

  void Reset() {}  // No-op, kept for API compatibility

  // Returns the peak RSS (max resident set size) of the process
  int64_t GetPeakMemory() const { return GetMaxRSS(); }

  void ReportToState(::benchmark::State& st) const { ReportSize(st, "peak_memory", GetPeakMemory()); }
};

//=============================================================================
// Thread Count Tracking Utilities
//=============================================================================

// Get current thread count for this process
inline int GetCurrentThreadCount() {
#ifdef __APPLE__
  mach_port_t task = mach_task_self();
  thread_act_array_t threads;
  mach_msg_type_number_t thread_count;
  if (task_threads(task, &threads, &thread_count) == KERN_SUCCESS) {
    vm_deallocate(task, reinterpret_cast<vm_address_t>(threads), thread_count * sizeof(thread_act_t));
    return static_cast<int>(thread_count);
  }
  return -1;
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return std::stoi(line.substr(8));
    }
  }
  return -1;
#else
  return -1;
#endif
}

class ThreadTracker {
  public:
  ThreadTracker() : running_(false), peak_threads_(0) {}

  ~ThreadTracker() { Stop(); }

  void Start(std::chrono::milliseconds interval = std::chrono::milliseconds(1)) {
    if (running_.exchange(true)) {
      return;
    }
    peak_threads_ = GetCurrentThreadCount();
    sampler_thread_ = std::thread([this, interval]() {
      while (running_.load()) {
        int current = GetCurrentThreadCount();
        if (current > 0) {
          int expected = peak_threads_.load();
          while (current > expected && !peak_threads_.compare_exchange_weak(expected, current)) {
          }
        }
        std::this_thread::sleep_for(interval);
      }
    });
  }

  void Stop() {
    if (running_.exchange(false)) {
      if (sampler_thread_.joinable()) {
        sampler_thread_.join();
      }
    }
  }

  int GetPeakThreads() const { return peak_threads_.load(); }

  void ReportToState(::benchmark::State& st) const {
    st.counters["peak_threads"] =
        ::benchmark::Counter(static_cast<double>(GetPeakThreads()), ::benchmark::Counter::kDefaults);
  }

  private:
  std::atomic<bool> running_;
  std::atomic<int> peak_threads_;
  std::thread sampler_thread_;
};

//=============================================================================
// Format Benchmark Fixture Base Class
//=============================================================================

template <bool EnableMemoryTracker = true, bool EnableThreadTracker = false>
class FormatBenchFixtureBase : public ::benchmark::Fixture {
  public:
  void SetUp(::benchmark::State& st) override {
    // Initialize properties from environment
    BENCH_ASSERT_STATUS_OK(InitTestProperties(properties_), st);

    // Configure global properties for benchmarks
    ConfigureGlobalProp();

    // Get filesystem
    BENCH_ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_), st);

    // Setup base path
    base_path_ = GetTestBasePath("format_benchmark");
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_), st);

    // Get available formats
    available_formats_ = GetAvailableFormats();

    // Initialize data loader from environment
    // If CUSTOM_SEGMENT_PATH is set, use MilvusSegmentLoader; otherwise use SyntheticDataLoader
    data_loader_ = CreateDataLoaderFromEnv(SyntheticDataConfig::Medium());
    auto load_status = data_loader_->Load();
    if (!load_status.ok()) {
      st.SkipWithError(("Failed to load benchmark data: " + load_status.ToString()).c_str());
      return;
    }

    // Initialize trackers if enabled
    if constexpr (EnableMemoryTracker) {
      memory_tracker_.Reset();
    }
    if constexpr (EnableThreadTracker) {
      thread_tracker_.Start();
    }
  }

  void TearDown(::benchmark::State& st) override {
    // Stop and report thread tracker if enabled
    if constexpr (EnableThreadTracker) {
      thread_tracker_.Stop();
      thread_tracker_.ReportToState(st);
    }

    // Report memory metrics if enabled
    if constexpr (EnableMemoryTracker) {
      memory_tracker_.ReportToState(st);
    }

    // Release data loader to free loaded data
    data_loader_.reset();

    // Clean up test directory
    auto status = DeleteTestDir(fs_, base_path_);
    if (!status.ok()) {
      // Log but don't fail on cleanup errors
    }
  }

  protected:
  // Configure global properties for all benchmarks
  void ConfigureGlobalProp() { api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "32768"); }

  // Configure memory settings based on MemoryConfig
  void ConfigureMemory(const MemoryConfig& config) {
    api::SetValue(properties_, PROPERTY_WRITER_BUFFER_SIZE, std::to_string(config.buffer_size).c_str());
    api::SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_ROWS, std::to_string(config.batch_size).c_str());
    api::SetValue(properties_, PROPERTY_READER_RECORD_BATCH_MAX_SIZE, std::to_string(config.buffer_size).c_str());
  }

  //-----------------------------------------------------------------------
  // Data Loader Methods - Use data from configured source
  //-----------------------------------------------------------------------

  // Get data loader
  BenchmarkDataLoader* GetDataLoader() const { return data_loader_.get(); }

  // Get schema from data loader
  std::shared_ptr<arrow::Schema> GetLoaderSchema() const { return data_loader_->GetSchema(); }

  // Get record batch reader from data loader (for streaming reads, preferred for large datasets)
  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetLoaderBatchReader() const {
    return data_loader_->GetRecordBatchReader();
  }

  // Get single record batch from data loader (loads all data into memory)
  // Use GetLoaderBatchReader() for large datasets
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetLoaderBatch() const { return data_loader_->GetRecordBatch(); }

  // Get schema-based patterns for Writer policy
  std::string GetSchemaBasePatterns() const { return data_loader_->GetSchemaBasePatterns(); }

  // Get scalar projection columns
  std::shared_ptr<std::vector<std::string>> GetScalarProjection() const { return data_loader_->GetScalarProjection(); }

  // Get vector projection columns
  std::shared_ptr<std::vector<std::string>> GetVectorProjection() const { return data_loader_->GetVectorProjection(); }

  // Get number of rows in loaded data
  int64_t GetLoaderNumRows() const { return data_loader_->NumRows(); }

  // Get data description for labels
  std::string GetDataDescription() const { return data_loader_->GetDescription(); }

  //-----------------------------------------------------------------------
  // Legacy Methods - For backward compatibility (uses synthetic data)
  //-----------------------------------------------------------------------

  // Create schema for test data (legacy - uses CreateTestSchema directly)
  arrow::Result<std::shared_ptr<arrow::Schema>> CreateSchema(std::array<bool, 4> needed_columns = {true, true, true,
                                                                                                   true}) {
    return CreateTestSchema(needed_columns);
  }

  // Create test data batch (legacy - uses CreateTestData directly)
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateBatch(std::shared_ptr<arrow::Schema> schema,
                                                                 const DataSizeConfig& config,
                                                                 bool random_data = true,
                                                                 int64_t start_offset = 0) {
    return CreateTestData(schema, start_offset, random_data, config.num_rows, config.vector_dim, config.string_length);
  }

  // Create policy for specific format
  arrow::Result<std::unique_ptr<api::ColumnGroupPolicy>> CreatePolicyForFormat(
      const std::string& format, const std::shared_ptr<arrow::Schema>& schema) {
    return CreateSinglePolicy(format, schema);
  }

  // Get unique path for this benchmark iteration
  std::string GetUniquePath(const std::string& suffix = "") const {
    static std::atomic<uint64_t> counter{0};
    std::string path = base_path_ + "/bench_" + std::to_string(counter++);
    if (!suffix.empty()) {
      path += "_" + suffix;
    }
    return path;
  }

  // Check if format is available and skip if not
  bool CheckFormatAvailable(::benchmark::State& st, const std::string& format) {
    if (!IsFormatAvailable(format)) {
      st.SkipWithError(("Format not available: " + format).c_str());
      return false;
    }
    return true;
  }

  // Calculate logical data size for a batch.
  // Computes size from type layout and num_rows, consistent regardless of slicing.
  static int64_t CalculateRawDataSize(const std::shared_ptr<arrow::RecordBatch>& batch) {
    int64_t size = 0;
    int64_t num_rows = batch->num_rows();
    for (int i = 0; i < batch->num_columns(); ++i) {
      auto type = batch->column(i)->type();
      if (type->id() == arrow::Type::LIST) {
        // list<T>: use offsets to get actual child element count
        auto list_array = std::static_pointer_cast<arrow::ListArray>(batch->column(i));
        int64_t total_values = list_array->value_offset(num_rows) - list_array->value_offset(0);
        size += total_values * list_array->value_type()->byte_width();
      } else if (arrow::is_fixed_width(*type)) {
        size += num_rows * type->byte_width();
      } else if (type->id() == arrow::Type::STRING || type->id() == arrow::Type::BINARY) {
        // Variable-width (string/binary): use offsets
        auto offsets = batch->column(i)->data()->buffers[1];
        auto offset_base = batch->column(i)->offset();
        auto off_ptr = reinterpret_cast<const int32_t*>(offsets->data());
        size += off_ptr[offset_base + num_rows] - off_ptr[offset_base];
      } else {
        throw std::runtime_error("CalculateRawDataSize: unsupported type " + type->ToString());
      }
    }
    return size;
  }

  protected:
  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
  std::vector<std::string> available_formats_;
  std::unique_ptr<BenchmarkDataLoader> data_loader_;
  MemoryTracker memory_tracker_;
  ThreadTracker thread_tracker_;
};

}  // namespace benchmark
}  // namespace milvus_storage
