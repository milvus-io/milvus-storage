// Copyright 2026 Zilliz
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

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <parquet/properties.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/parquet/parquet_writer.h"

// Compression-level benchmark.
//
// All runs go through milvus_storage::parquet::ParquetFileWriter::Make,
// which hardcodes UNCOMPRESSED for vector columns (FIXED_SIZE_BINARY /
// BINARY). The only thing varied across configs is the file-level ZSTD
// compression level applied to non-vector columns.
//
// Schema:
//   pk:     int64               (sequential)
//   text:   utf8                (random 100-500 byte printable ASCII)
//   vector: fixed_size_binary   (truly random uint8 bytes, 768 * 4 = 3 KiB)
//
// Run:
//   ./benchmark --benchmark_filter=CompressionBench
namespace milvus_storage::benchmark_compression {

namespace {

constexpr int kVectorDim = 768;
constexpr int kRowsPerBatch = 1000;
constexpr int kBatches = 100;  // 100k rows total ≈ 294 MiB raw vector + scalars

struct WriterConfig {
  std::string label;
  int level;
};

const std::vector<WriterConfig>& Configs() {
  static const std::vector<WriterConfig> kConfigs = {
      {"zstd3", 3},
      {"zstd5", 5},
  };
  return kConfigs;
}

std::shared_ptr<arrow::Schema> MakeSchema() {
  return arrow::schema({
      arrow::field("pk", arrow::int64(), /*nullable=*/false),
      arrow::field("text", arrow::utf8(), /*nullable=*/false),
      arrow::field("vector", arrow::fixed_size_binary(kVectorDim * 4), /*nullable=*/false),
  });
}

// Generate deterministic batches so every config sees identical bytes.
std::vector<std::shared_ptr<arrow::RecordBatch>> MakeBatches(const std::shared_ptr<arrow::Schema>& schema) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> out;
  out.reserve(kBatches);

  std::mt19937 rng(0xC0DEC0DE);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> printable_dist(0x20, 0x7E);
  std::uniform_int_distribution<int> text_len_dist(100, 500);

  for (int b = 0; b < kBatches; ++b) {
    arrow::Int64Builder pk_builder;
    arrow::StringBuilder text_builder;
    arrow::FixedSizeBinaryBuilder vec_builder(arrow::fixed_size_binary(kVectorDim * 4));
    pk_builder.Resize(kRowsPerBatch).ok();
    vec_builder.Resize(kRowsPerBatch).ok();

    std::vector<uint8_t> vec_bytes(kVectorDim * 4);
    std::string text_buf;
    text_buf.reserve(500);
    for (int r = 0; r < kRowsPerBatch; ++r) {
      pk_builder.UnsafeAppend(static_cast<int64_t>(b * kRowsPerBatch + r));

      int text_len = text_len_dist(rng);
      text_buf.resize(text_len);
      for (int i = 0; i < text_len; ++i) {
        text_buf[i] = static_cast<char>(printable_dist(rng));
      }
      text_builder.Append(text_buf).ok();

      for (size_t i = 0; i < vec_bytes.size(); ++i) {
        vec_bytes[i] = static_cast<uint8_t>(byte_dist(rng));
      }
      vec_builder.UnsafeAppend(vec_bytes.data());
    }

    auto pk_arr = pk_builder.Finish().ValueOrDie();
    auto text_arr = text_builder.Finish().ValueOrDie();
    auto vec_arr = vec_builder.Finish().ValueOrDie();
    out.push_back(arrow::RecordBatch::Make(schema, kRowsPerBatch, {pk_arr, text_arr, vec_arr}));
  }
  return out;
}

std::shared_ptr<::parquet::WriterProperties> BuildWriterProps(const WriterConfig& cfg) {
  return ::parquet::WriterProperties::Builder()
      .compression(::parquet::Compression::ZSTD)
      ->compression_level(cfg.level)
      ->build();
}

class CompressionBench : public ::benchmark::Fixture {
  public:
  void SetUp(::benchmark::State& st) override {
    schema_ = MakeSchema();
    if (batches_.empty()) {
      batches_ = MakeBatches(schema_);
    }
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
    base_path_ = "/tmp/milvus_storage_compression_bench";
    auto _ = fs_->CreateDir(base_path_, /*recursive=*/true);
    (void)_;
  }

  void TearDown(::benchmark::State&) override {}

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  static std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::shared_ptr<arrow::fs::LocalFileSystem> fs_;
  std::string base_path_;
};

std::vector<std::shared_ptr<arrow::RecordBatch>> CompressionBench::batches_;

}  // namespace

BENCHMARK_DEFINE_F(CompressionBench, Write)(::benchmark::State& st) {
  const auto& cfg = Configs()[static_cast<size_t>(st.range(0))];
  st.SetLabel(cfg.label);
  auto writer_props = BuildWriterProps(cfg);

  int64_t last_file_size = 0;
  int64_t iter_idx = 0;

  milvus_storage::StorageConfig storage_config;
  for (auto _ : st) {
    auto file_path = base_path_ + "/" + cfg.label + "_" + std::to_string(iter_idx++) + ".parquet";

    auto writer_res =
        milvus_storage::parquet::ParquetFileWriter::Make(schema_, fs_, file_path, storage_config, writer_props);
    if (!writer_res.ok()) {
      st.SkipWithError(writer_res.status().ToString().c_str());
      return;
    }
    auto writer = std::move(writer_res).ValueOrDie();

    for (const auto& batch : batches_) {
      auto status = writer->Write(batch);
      if (!status.ok()) {
        st.SkipWithError(status.ToString().c_str());
        return;
      }
    }

    auto close_res = writer->Close();
    if (!close_res.ok()) {
      st.SkipWithError(close_res.status().ToString().c_str());
      return;
    }

    auto info = fs_->GetFileInfo(file_path);
    if (info.ok()) {
      last_file_size = info.ValueOrDie().size();
    }
    auto del = fs_->DeleteFile(file_path);
    (void)del;
  }

  int64_t total_rows = static_cast<int64_t>(kBatches) * kRowsPerBatch;
  // text mean length ≈ 300 bytes (uniform 100..500). pk is int64 (8 B);
  // vector is fixed_size_binary(kVectorDim*4).
  int64_t raw_bytes = total_rows * (8 + 300 + kVectorDim * 4);
  st.counters["file_MiB"] = ::benchmark::Counter(static_cast<double>(last_file_size) / 1048576.0);
  st.counters["raw_MiB_approx"] = ::benchmark::Counter(static_cast<double>(raw_bytes) / 1048576.0);
  st.counters["rows"] = ::benchmark::Counter(static_cast<double>(total_rows));
}

BENCHMARK_REGISTER_F(CompressionBench, Write)
    ->Arg(0)  // zstd3
    ->Arg(1)  // zstd5
    ->Unit(::benchmark::kMillisecond)
    ->Iterations(3);

}  // namespace milvus_storage::benchmark_compression
