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

#include <benchmark/benchmark.h>

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <random>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/api.h>
#include <parquet/properties.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>

#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/io/api.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/writer.h"
#include "test_env.h"

#define GBENCH_ASSERT_STATUS_OK(status, st)            \
  do {                                                 \
    if (!(status).ok()) {                              \
      (st).SkipWithError((status).ToString().c_str()); \
    }                                                  \
  } while (false)

#define GBENCH_ASSERT_AND_ASSIGN_IMPL(status_name, lhs, rexpr, st) \
  auto status_name = (rexpr);                                      \
  GBENCH_ASSERT_STATUS_OK(status_name.status(), st);               \
  lhs = std::move(status_name).ValueOrDie();

#define GBENCH_ASSERT_AND_ASSIGN(lhs, rexpr, st) \
  GBENCH_ASSERT_AND_ASSIGN_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr, st);

namespace milvus_storage {
using milvus_storage::api::ColumnGroupPolicy;
using milvus_storage::api::Writer;

// Helper function to create a schema with int64, bool, vector, and string
arrow::Result<std::shared_ptr<arrow::Schema>> CreateFooterBenchmarkSchema(size_t vector_dim) {
  std::vector<std::shared_ptr<arrow::Field>> fields;

  fields.emplace_back(
      arrow::field("id", arrow::int64(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"1"})));
  fields.emplace_back(
      arrow::field("flag", arrow::boolean(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"2"})));
  // Vector stored as fixed_size_binary: 768d float32 = 768 * 4 = 3072 bytes
  fields.emplace_back(arrow::field("vector", arrow::fixed_size_binary(vector_dim * sizeof(float)), false,
                                   arrow::key_value_metadata({"PARQUET:field_id"}, {"3"})));
  fields.emplace_back(
      arrow::field("text", arrow::utf8(), false, arrow::key_value_metadata({"PARQUET:field_id"}, {"4"})));

  return arrow::schema(fields);
}

// Helper function to create test data
arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateFooterBenchmarkData(
    const std::shared_ptr<arrow::Schema>& schema, size_t num_rows, size_t vector_dim, size_t string_length) {
  arrow::Int64Builder id_builder;
  arrow::BooleanBuilder bool_builder;
  // Vector stored as fixed_size_binary: vector_dim * sizeof(float) bytes
  arrow::FixedSizeBinaryBuilder vector_builder(arrow::fixed_size_binary(vector_dim * sizeof(float)));
  arrow::StringBuilder string_builder;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> id_dist(0, 1000000);
  std::uniform_int_distribution<int> bool_dist(0, 1);
  std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> char_dist(33, 126);

  // Pre-allocate vector buffer
  const size_t vector_bytes = vector_dim * sizeof(float);
  std::vector<uint8_t> vector_buffer(vector_bytes);

  for (size_t i = 0; i < num_rows; ++i) {
    // int64
    ARROW_RETURN_NOT_OK(id_builder.Append(id_dist(gen)));

    // bool
    ARROW_RETURN_NOT_OK(bool_builder.Append(bool_dist(gen) == 1));

    // vector (fixed_size_binary: vector_dim float32 values)
    auto* vector_data = reinterpret_cast<float*>(vector_buffer.data());
    for (size_t j = 0; j < vector_dim; ++j) {
      vector_data[j] = float_dist(gen);
    }
    ARROW_RETURN_NOT_OK(vector_builder.Append(vector_buffer.data()));

    // string
    std::string str;
    str.reserve(string_length);
    for (size_t j = 0; j < string_length; ++j) {
      str += static_cast<char>(char_dist(gen));
    }
    ARROW_RETURN_NOT_OK(string_builder.Append(str));
  }

  std::shared_ptr<arrow::Array> id_array, bool_array, vector_array, string_array;
  ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
  ARROW_RETURN_NOT_OK(bool_builder.Finish(&bool_array));
  ARROW_RETURN_NOT_OK(vector_builder.Finish(&vector_array));
  ARROW_RETURN_NOT_OK(string_builder.Finish(&string_array));

  return arrow::RecordBatch::Make(schema, num_rows, {id_array, bool_array, vector_array, string_array});
}

// Helper function to get footer size from parquet file
arrow::Result<int64_t> GetParquetFooterSize(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                            const std::string& file_path) {
  // Open the file
  ARROW_ASSIGN_OR_RAISE(auto file, fs->OpenInputFile(file_path));

  // Get file size
  ARROW_ASSIGN_OR_RAISE(auto file_size, file->GetSize());

  // Read the last 8 bytes to get footer length
  // Parquet format: [4 bytes footer length][4 bytes magic number "PAR1"]
  constexpr int64_t footer_length_size = 4;
  constexpr int64_t magic_size = 4;
  constexpr int64_t footer_info_size = footer_length_size + magic_size;

  ARROW_ASSIGN_OR_RAISE(auto footer_info_buffer, file->ReadAt(file_size - footer_info_size, footer_info_size));

  // Extract footer length (little-endian)
  const uint8_t* data = footer_info_buffer->data();
  uint32_t footer_length = *reinterpret_cast<const uint32_t*>(data);

  // Verify magic number
  const char* magic = reinterpret_cast<const char*>(data + footer_length_size);
  if (std::string(magic, magic_size) != "PAR1") {
    return arrow::Status::Invalid("Invalid parquet magic number");
  }

  // Footer size = footer length + 8 bytes (4 for length + 4 for magic)
  return static_cast<int64_t>(footer_length) + footer_info_size;
}

class FooterSizeFixture : public benchmark::Fixture {
  protected:
  void SetUp(::benchmark::State& state) override {
    GBENCH_ASSERT_STATUS_OK(InitTestProperties(properties_), state);
    GBENCH_ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_), state);
    base_path_ = GetTestBasePath("footer_benchmark");
    GBENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_), state);
    GBENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_), state);
  }

  void TearDown(::benchmark::State& state) override {
    // Cleanup if needed
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

BENCHMARK_DEFINE_F(FooterSizeFixture, MeasureFooterSize)(benchmark::State& st) {
  auto num_rows = static_cast<size_t>(st.range(0));
  auto vector_dim = static_cast<size_t>(st.range(1));
  auto string_length = static_cast<size_t>(st.range(2));

  GBENCH_ASSERT_AND_ASSIGN(auto schema, CreateFooterBenchmarkSchema(vector_dim), st);
  GBENCH_ASSERT_AND_ASSIGN(auto record_batch, CreateFooterBenchmarkData(schema, num_rows, vector_dim, string_length),
                           st);

  for (auto _ : st) {
    // Write data
    GBENCH_ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(properties_, schema), st);
    auto writer = Writer::create(base_path_, schema, std::move(policy), properties_);
    GBENCH_ASSERT_STATUS_OK(writer->write(record_batch), st);
    GBENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);

    // Get the parquet file path from column groups
    // Assuming single column group with one file
    auto all_column_groups = cgs->get_all();
    if (!all_column_groups.empty() && !all_column_groups[0]->files.empty()) {
      // Path from column groups already includes the base_path
      std::string parquet_file_path = all_column_groups[0]->files[0].path;

      // Measure footer size
      GBENCH_ASSERT_AND_ASSIGN(auto footer_size, GetParquetFooterSize(fs_, parquet_file_path), st);

      // Report footer size
      st.counters["footer_size_bytes"] =
          benchmark::Counter(static_cast<double>(footer_size), benchmark::Counter::kAvgIterations);

      // Also report file size for context
      GBENCH_ASSERT_AND_ASSIGN(auto file_size, fs_->GetFileInfo(parquet_file_path), st);
      st.counters["file_size_bytes"] =
          benchmark::Counter(static_cast<double>(file_size.size()), benchmark::Counter::kAvgIterations);

      // Report footer as percentage of file size
      if (file_size.size() > 0) {
        double footer_percentage = (static_cast<double>(footer_size) / static_cast<double>(file_size.size())) * 100.0;
        st.counters["footer_percentage"] = benchmark::Counter(footer_percentage, benchmark::Counter::kAvgIterations);
      }
    }
  }
}

BENCHMARK_REGISTER_F(FooterSizeFixture, MeasureFooterSize)
    ->Args({1000, 128, 64})     // 1K rows, 128-dim vector, 64-char strings
    ->Args({10000, 128, 64})    // 10K rows, 128-dim vector, 64-char strings
    ->Args({100000, 128, 64})   // 100K rows, 128-dim vector, 64-char strings
    ->Args({1000, 768, 128})    // 1K rows, 768-dim vector, 128-char strings
    ->Args({10000, 768, 128})   // 10K rows, 768-dim vector, 128-char strings
    ->Args({100000, 768, 128})  // 100K rows, 768-dim vector, 128-char strings
    ->Iterations(1)
    ->Unit(benchmark::kMicrosecond);

}  // namespace milvus_storage
