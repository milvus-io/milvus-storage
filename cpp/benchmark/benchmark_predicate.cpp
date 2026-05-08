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

#include <iomanip>
#include <sstream>
#include <thread>
#include <arrow/table.h>
#include "milvus-storage/thread_pool.h"
#include "milvus-storage/filesystem/observable.h"

namespace milvus_storage::benchmark {

using namespace milvus_storage::api;

//=============================================================================
// Predicate Pushdown Benchmark
//
// Args: [sorted, selectivity_pct, predicate_col]
//   sorted: 0 = unsorted (random), 1 = sorted (sequential)
//   selectivity_pct: 0 = no predicate, 10 = keep 10%, 50 = keep 50%, 90 = keep 90%
//   predicate_col: 0 = int64 (id), 1 = string (name)
//=============================================================================

/// Zero-pad an integer to a fixed width string
static std::string ZeroPad(int64_t val, int width) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << val;
  return oss.str();
}

class PredicateBenchmark : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);
    schema_ = GetLoaderSchema();
  }

  void TearDown(::benchmark::State& st) override {
    schema_.reset();
    ThreadPoolHolder::Release();
    FormatBenchFixtureBase<>::TearDown(st);
  }

  protected:
  /// Build a RecordBatch with zero-padded names.
  /// Both sorted and unsorted use the SAME values (id=0..N-1, name="name_00000"..),
  /// so predicates have identical selectivity. The only difference is row order:
  /// sorted = sequential, unsorted = shuffled. This isolates the effect of
  /// data ordering (zone-map pruning) from predicate selectivity.
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreatePaddedBatch(const std::shared_ptr<arrow::Schema>& schema,
                                                                       const DataSizeConfig& config,
                                                                       bool sorted) {
    int64_t num_rows = static_cast<int64_t>(config.num_rows);
    int pad_width = static_cast<int>(std::to_string(num_rows - 1).size());

    // Generate row indices: 0..N-1, then optionally shuffle
    std::vector<int64_t> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    if (!sorted) {
      std::mt19937 rng(42);  // fixed seed for reproducibility
      std::shuffle(indices.begin(), indices.end(), rng);
    }

    arrow::Int64Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::DoubleBuilder value_builder;
    arrow::ListBuilder vector_builder(arrow::default_memory_pool(), std::make_shared<arrow::FloatBuilder>());
    auto& float_builder = *static_cast<arrow::FloatBuilder*>(vector_builder.value_builder());

    ARROW_RETURN_NOT_OK(id_builder.Reserve(num_rows));
    ARROW_RETURN_NOT_OK(name_builder.Reserve(num_rows));
    ARROW_RETURN_NOT_OK(value_builder.Reserve(num_rows));

    for (int64_t row = 0; row < num_rows; ++row) {
      int64_t i = indices[row];
      ARROW_RETURN_NOT_OK(id_builder.Append(i));
      ARROW_RETURN_NOT_OK(name_builder.Append("name_" + ZeroPad(i, pad_width)));
      ARROW_RETURN_NOT_OK(value_builder.Append(i * 1.5));

      ARROW_RETURN_NOT_OK(vector_builder.Append());
      for (size_t d = 0; d < config.vector_dim; ++d) {
        ARROW_RETURN_NOT_OK(float_builder.Append(static_cast<float>(i * config.vector_dim + d)));
      }
    }

    std::shared_ptr<arrow::Array> id_arr, name_arr, value_arr, vector_arr;
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_arr));
    ARROW_RETURN_NOT_OK(name_builder.Finish(&name_arr));
    ARROW_RETURN_NOT_OK(value_builder.Finish(&value_arr));
    ARROW_RETURN_NOT_OK(vector_builder.Finish(&vector_arr));

    return arrow::RecordBatch::Make(schema, num_rows, {id_arr, name_arr, value_arr, vector_arr});
  }

  arrow::Status PrepareTestData(const std::string& format,
                                bool sorted,
                                std::shared_ptr<ColumnGroups>& out_cgs,
                                std::string& out_path,
                                int64_t& out_num_rows) {
    out_path = GetUniquePath(format + "_predicate_test");

    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));

    auto writer = Writer::create(out_path, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create writer");
    }

    auto data_config = DataSizeConfig::Medium();
    ARROW_ASSIGN_OR_RAISE(auto batch, CreatePaddedBatch(schema_, data_config, sorted));
    out_num_rows = batch->num_rows();
    ARROW_RETURN_NOT_OK(writer->write(batch));
    ARROW_ASSIGN_OR_RAISE(out_cgs, writer->close());
    return arrow::Status::OK();
  }

  /// Build predicate string. For sorted data with padded names,
  /// lexicographic and numeric order match.
  static std::string BuildPredicate(int selectivity_pct, int predicate_col, int64_t num_rows) {
    if (selectivity_pct <= 0 || selectivity_pct >= 100) {
      return "";
    }

    int64_t threshold = num_rows * (100 - selectivity_pct) / 100;

    if (predicate_col == 0) {
      return "id > " + std::to_string(threshold);
    } else {
      int pad_width = static_cast<int>(std::to_string(num_rows - 1).size());
      return "name > 'name_" + ZeroPad(threshold, pad_width) + "'";
    }
  }

  std::shared_ptr<FilesystemMetrics> GetFsMetrics() {
    auto observable = std::dynamic_pointer_cast<Observable>(fs_);
    return observable ? observable->GetMetrics() : nullptr;
  }

  std::shared_ptr<arrow::Schema> schema_;
};

// Args: [sorted, selectivity_pct, predicate_col]
BENCHMARK_DEFINE_F(PredicateBenchmark, ReadWithPredicate)(::benchmark::State& st) {
  bool sorted = st.range(0) != 0;
  int selectivity_pct = static_cast<int>(st.range(1));
  int predicate_col = static_cast<int>(st.range(2));

  std::string format = LOON_FORMAT_VORTEX;
  if (!CheckFormatAvailable(st, format)) {
    return;
  }

  MemoryConfig memory_config = MemoryConfig::Default();
  ConfigureMemory(memory_config);
  ThreadPoolHolder::WithSingleton(1);

  std::shared_ptr<ColumnGroups> cgs;
  std::string path;
  int64_t num_rows = 0;
  BENCH_ASSERT_STATUS_OK(PrepareTestData(format, sorted, cgs, path, num_rows), st);

  std::string predicate = BuildPredicate(selectivity_pct, predicate_col, num_rows);
  auto fs_metrics = GetFsMetrics();

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;
  int64_t total_io_bytes = 0;
  int64_t total_io_count = 0;

  for (auto _ : st) {
    if (fs_metrics) {
      fs_metrics->Reset();
    }

    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    if (predicate.empty()) {
      BENCH_ASSERT_AND_ASSIGN(batch_reader, reader->get_record_batch_reader(), st);
    } else {
      BENCH_ASSERT_AND_ASSIGN(batch_reader, reader->get_record_batch_reader(predicate), st);
    }

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader->ReadNext(&batch), st);
      if (!batch)
        break;
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }

    if (fs_metrics) {
      total_io_bytes += fs_metrics->GetReadBytes();
      total_io_count += fs_metrics->GetReadCount();
    }
  }

  // Report metrics
  ReportThroughput(st, total_bytes_read, total_rows_read);

  double iters = static_cast<double>(st.iterations());

  // Selectivity
  double actual_selectivity = (iters > 0 && num_rows > 0)
                                  ? static_cast<double>(total_rows_read) / (iters * static_cast<double>(num_rows))
                                  : 0.0;
  st.counters["selectivity"] = ::benchmark::Counter(actual_selectivity, ::benchmark::Counter::kDefaults);

  // I/O metrics
  if (total_io_bytes > 0) {
    st.counters["io_bytes_per_iter"] =
        ::benchmark::Counter(static_cast<double>(total_io_bytes) / iters, ::benchmark::Counter::kDefaults);
    st.counters["io_count_per_iter"] =
        ::benchmark::Counter(static_cast<double>(total_io_count) / iters, ::benchmark::Counter::kDefaults);
  }

  // CPU time ratio (CPU time / wall time) — measures CPU utilization
  // Google Benchmark reports both Time (wall) and CPU columns;
  // cpu_time_ratio > 1.0 means multi-threaded CPU usage.
  // We don't need to compute it manually — the Time and CPU columns suffice.

  // Label
  std::string col_name = (predicate_col == 0) ? "int64" : "string";
  std::string label = format + "/" + (sorted ? "sorted" : "unsorted") + "/";
  if (selectivity_pct == 0) {
    label += "no_predicate";
  } else {
    label += col_name + "_" + std::to_string(selectivity_pct) + "pct";
  }
  label += "/" + GetDataDescription();
  st.SetLabel(label);
}

BENCHMARK_REGISTER_F(PredicateBenchmark, ReadWithPredicate)
    ->ArgsProduct({
        {0, 1},           // sorted: unsorted(0), sorted(1)
        {0, 10, 50, 90},  // selectivity: no_predicate(0), 10%, 50%, 90%
        {0}               // predicate_col: int64(0)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(PredicateBenchmark, ReadWithPredicate)
    ->ArgsProduct({
        {0, 1},        // sorted: unsorted(0), sorted(1)
        {10, 50, 90},  // selectivity: 10%, 50%, 90%
        {1}            // predicate_col: string(1)
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace milvus_storage::benchmark
