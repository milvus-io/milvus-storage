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

// V2 vs V3 Reader Benchmark
// Compare performance between packed/ low-level reader (V2: PackedRecordBatchReader)
// and top-level Reader API (V3: get_record_batch_reader / get_chunk_reader).
// V2 uses PackedRecordBatchWriter + PackedRecordBatchReader directly.
// V3 uses the high-level Writer + Reader API.

#include "benchmark_format_common.h"

#include <arrow/table.h>

#include "milvus-storage/packed/writer.h"
#include "milvus-storage/packed/reader.h"
#include "milvus-storage/thread_pool.h"

namespace milvus_storage {
namespace benchmark {

using namespace milvus_storage::api;

//=============================================================================
// V2 vs V3 Benchmark Fixture
//=============================================================================

inline size_t ComputeNumBatches(size_t num_rows) {
  return std::max<size_t>(1, num_rows / 1000);  // ~1000 rows per batch
}

class V2V3BenchFixture : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);

    if (!CheckFormatAvailable(st, LOON_FORMAT_PARQUET)) {
      return;
    }

    // Get schema from data loader
    schema_ = GetLoaderSchema();

    // Pre-load all batches for write benchmarks
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

    ThreadPoolHolder::WithSingleton(1);
  }

  void TearDown(::benchmark::State& st) override {
    // Clear pre-loaded batches to release memory
    batches_.clear();
    batches_.shrink_to_fit();
    schema_.reset();
    ThreadPoolHolder::Release();
    FormatBenchFixtureBase<>::TearDown(st);
  }

  protected:
  // Write data using PackedRecordBatchWriter (V2 path) and return the file path
  arrow::Status PrepareV2Data(std::string& out_path) {
    out_path = GetUniquePath("v2_test") + "/data.parquet";
    std::vector<std::string> paths = {out_path};

    // Build column groups based on schema
    std::vector<std::vector<int>> column_groups;
    std::vector<int> all_cols;
    for (int i = 0; i < schema_->num_fields(); ++i) {
      all_cols.push_back(i);
    }
    column_groups.push_back(all_cols);

    StorageConfig storage_config;

    ARROW_ASSIGN_OR_RAISE(auto writer, PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config, column_groups,
                                                                     DEFAULT_WRITE_BUFFER_SIZE));

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      ARROW_RETURN_NOT_OK(writer->Write(batch));
    }
    auto result = writer->Close();

    return arrow::Status::OK();
  }

  // Write data using Writer API (V3 path) and return column groups
  arrow::Status PrepareV3Data(std::shared_ptr<ColumnGroups>& out_cgs) {
    std::string path = GetUniquePath("v3_test");

    // Use schema-based policy
    std::string patterns = GetSchemaBasePatterns();
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSchemaBasePolicy(patterns, LOON_FORMAT_PARQUET, schema_));

    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create writer");
    }

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(out_cgs, writer->close());

    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  int64_t total_bytes_ = 0;
  int64_t total_rows_ = 0;
};

//=============================================================================
// V2: PackedRecordBatchReader Benchmark (Low-level)
//=============================================================================

BENCHMARK_DEFINE_F(V2V3BenchFixture, V2_PackedRecordBatchReader)(::benchmark::State& st) {
  // Write data using packed writer
  std::string path;
  BENCH_ASSERT_STATUS_OK(PrepareV2Data(path), st);

  std::vector<std::string> paths = {path};

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    PackedRecordBatchReader reader(fs_, paths, schema_, DEFAULT_READ_BUFFER_SIZE);

    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(reader.ReadNext(&batch), st);
      if (batch == nullptr)
        break;
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
    BENCH_ASSERT_STATUS_OK(reader.Close(), st);
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.SetLabel("v2/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchReader)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// V3: Reader::get_record_batch_reader Benchmark (Top-level)
//=============================================================================

BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_RecordBatchReader)(::benchmark::State& st) {
  // Write data using Writer API
  std::shared_ptr<ColumnGroups> cgs;
  BENCH_ASSERT_STATUS_OK(PrepareV3Data(cgs), st);

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
  st.SetLabel("v3-rb/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_RecordBatchReader)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// V3: Reader::get_chunk_reader Benchmark (Top-level Chunk Access)
//=============================================================================

// Args: [config_idx]
BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_ChunkReader)(::benchmark::State& st) {
  // Write data using Writer API
  std::shared_ptr<ColumnGroups> cgs;
  BENCH_ASSERT_STATUS_OK(PrepareV3Data(cgs), st);

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    BENCH_ASSERT_AND_ASSIGN(auto chunk_reader, reader->get_chunk_reader(0), st);

    size_t total_chunks = chunk_reader->total_number_of_chunks();
    for (size_t i = 0; i < total_chunks; ++i) {
      BENCH_ASSERT_AND_ASSIGN(auto batch, chunk_reader->get_chunk(i), st);
      total_rows_read += batch->num_rows();
      total_bytes_read += CalculateRawDataSize(batch);
    }
  }

  ReportThroughput(st, total_bytes_read, total_rows_read);
  st.SetLabel("v3-chunk/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_ChunkReader)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// V2: PackedRecordBatchWriter Benchmark (Low-level)
//=============================================================================

BENCHMARK_DEFINE_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)(::benchmark::State& st) {
  std::string base_path = GetUniquePath("v2_write_bench");

  // Build column groups based on schema
  std::vector<std::vector<int>> column_groups;
  std::vector<int> all_cols;
  for (int i = 0; i < schema_->num_fields(); ++i) {
    all_cols.push_back(i);
  }
  column_groups.push_back(all_cols);

  for (auto _ : st) {
    std::string path = base_path + "/data.parquet";
    std::vector<std::string> paths = {path};
    StorageConfig storage_config;

    BENCH_ASSERT_AND_ASSIGN(
        auto writer,
        PackedRecordBatchWriter::Make(fs_, paths, schema_, storage_config, column_groups, DEFAULT_WRITE_BUFFER_SIZE),
        st);

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      BENCH_ASSERT_STATUS_OK(writer->Write(batch), st);
    }
    auto result = writer->Close();

    // Cleanup for next iteration
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path), st);
  }

  int64_t total_bytes = total_bytes_ * static_cast<int64_t>(st.iterations());
  int64_t total_rows = total_rows_ * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  st.SetLabel("v2-writer/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// V3: Writer API Benchmark (Top-level)
//=============================================================================

BENCHMARK_DEFINE_F(V2V3BenchFixture, V3_Writer)(::benchmark::State& st) {
  std::string base_path = GetUniquePath("v3_write_bench");

  for (auto _ : st) {
    // Use schema-based policy
    std::string patterns = GetSchemaBasePatterns();
    BENCH_ASSERT_AND_ASSIGN(auto policy, CreateSchemaBasePolicy(patterns, LOON_FORMAT_PARQUET, schema_), st);
    auto writer = Writer::create(base_path, schema_, std::move(policy), properties_);
    BENCH_ASSERT_NOT_NULL(writer, st);

    // Write all pre-loaded batches
    for (const auto& batch : batches_) {
      BENCH_ASSERT_STATUS_OK(writer->write(batch), st);
    }
    BENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    // Cleanup for next iteration
    BENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path), st);
    BENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path), st);
  }

  int64_t total_bytes = total_bytes_ * static_cast<int64_t>(st.iterations());
  int64_t total_rows = total_rows_ * static_cast<int64_t>(st.iterations());
  ReportThroughput(st, total_bytes, total_rows);
  st.SetLabel("v3-writer/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_Writer)->Unit(::benchmark::kMillisecond)->UseRealTime();

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchReader)
    ->Name("Typical/V2_Reader")
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_RecordBatchReader)
    ->Name("Typical/V3_Reader")
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V2_PackedRecordBatchWriter)
    ->Name("Typical/V2_Writer")
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(V2V3BenchFixture, V3_Writer)
    ->Name("Typical/V3_Writer")
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace benchmark
}  // namespace milvus_storage
