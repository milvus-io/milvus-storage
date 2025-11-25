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
#include <ratio>
#include <string>
#include <vector>
#include <iostream>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/api.h>
#include <parquet/properties.h>

#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/util/key_value_metadata.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/reader.h"
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

#define GBENCH_ASSERT_NE(val1, val2, st)        \
  do {                                          \
    if ((val1) == (val2)) {                     \
      (st).SkipWithError("Values are equal: "); \
    }                                           \
  } while (false)

namespace milvus_storage {
using namespace milvus_storage::api;

class StorageFixture : public benchmark::Fixture {
  protected:
  void SetUp(::benchmark::State& state) override {
    GBENCH_ASSERT_STATUS_OK(InitTestProperties(properties_), state);
    GBENCH_ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_), state);
    base_path_ = GetTestBasePath("googlebench");
    GBENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_), state);
    GBENCH_ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_), state);
  }

  void TearDown(::benchmark::State& state) override {
    // GBENCH_ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_), state);
  }

  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

struct WriteDataConfig {
  size_t num_rows = 4096;
  size_t vector_dim = 128;
  size_t string_length = 128;
  bool random_data = true;
  std::array<bool, 4> needed_columns = {true, true, true, true};
};

static void APIWriteBenchmark(benchmark::State& st,
                              std::shared_ptr<arrow::fs::FileSystem> fs,
                              std::string& base_path,
                              const api::Properties& properties,
                              size_t loop_times,
                              const WriteDataConfig& write_config) {
  GBENCH_ASSERT_AND_ASSIGN(auto schema, CreateTestSchema(write_config.needed_columns), st);
  GBENCH_ASSERT_AND_ASSIGN(
      auto record_batch,
      CreateTestData(schema, 0, write_config.random_data, write_config.num_rows, write_config.vector_dim,
                     write_config.string_length, write_config.needed_columns),
      st);

  for (auto _ : st) {
    GBENCH_ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(properties, schema), st);
    auto writer = Writer::create(base_path, schema, std::move(policy), std::move(properties));
    for (size_t i = 0; i < loop_times; ++i) {
      GBENCH_ASSERT_STATUS_OK(writer->write(record_batch), st);
    }
    GBENCH_ASSERT_AND_ASSIGN(auto _unused, writer->close(), st);
  }
}

struct FullReadDataConfig {
  std::array<bool, 4> needed_columns = {true, true, true, true};
};

static std::shared_ptr<std::vector<std::string>> GetProjection(std::array<bool, 4> needed_columns) {
  std::vector<std::string> full_projection = {"id", "name", "value", "vector"};
  std::shared_ptr<std::vector<std::string>> projection = std::make_shared<std::vector<std::string>>();
  for (size_t i = 0; i < needed_columns.size(); ++i) {
    if (needed_columns[i]) {
      projection->emplace_back(full_projection[i]);
    }
  }

  return projection;
}

static void APIFullReadBenchmark(benchmark::State& st,
                                 std::shared_ptr<arrow::fs::FileSystem> fs,
                                 std::string& base_path,
                                 const api::Properties& properties,
                                 size_t loop_times,
                                 const WriteDataConfig& write_config,
                                 const FullReadDataConfig& read_config) {
  auto projection = GetProjection(read_config.needed_columns);
  GBENCH_ASSERT_AND_ASSIGN(auto schema, CreateTestSchema(write_config.needed_columns), st);
  GBENCH_ASSERT_AND_ASSIGN(
      auto record_batch,
      CreateTestData(schema, 0, write_config.random_data, write_config.num_rows, write_config.vector_dim,
                     write_config.string_length, write_config.needed_columns),
      st);

  GBENCH_ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(properties, schema), st);
  auto writer = Writer::create(base_path, schema, std::move(policy), std::move(properties));
  for (size_t i = 0; i < loop_times; ++i) {
    GBENCH_ASSERT_STATUS_OK(writer->write(record_batch), st);
  }
  GBENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);
  GBENCH_ASSERT_AND_ASSIGN(auto reader_schema, CreateTestSchema(read_config.needed_columns), st);
  for (auto _ : st) {
    auto reader = Reader::create(cgs, reader_schema, projection, properties);
    GBENCH_ASSERT_NE(reader, nullptr, st);

    for (size_t i = 0; i < loop_times; ++i) {
      // Test get_record_batch_reader (uses PackedRecordBatchReader internally)
      GBENCH_ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader(), st);
      while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        GBENCH_ASSERT_STATUS_OK(batch_reader->ReadNext(&batch), st);
        if (batch == nullptr) {
          break;
        }
      }
    }
  }
}

static void APIWriteLargeBenchmark(benchmark::State& st,
                                   std::shared_ptr<arrow::fs::FileSystem> fs,
                                   std::string& base_path,
                                   const api::Properties& properties,
                                   size_t target_row,
                                   size_t multi_random_rb_counts,
                                   const WriteDataConfig& write_config,
                                   const FullReadDataConfig& read_config) {
  auto projection = GetProjection(read_config.needed_columns);
  GBENCH_ASSERT_AND_ASSIGN(auto schema, CreateTestSchema(write_config.needed_columns), st);

  auto target_write_times = target_row / write_config.num_rows;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (size_t i = 0; i < multi_random_rb_counts; ++i) {
    GBENCH_ASSERT_AND_ASSIGN(
        auto batch,
        CreateTestData(schema, 0, write_config.random_data, write_config.num_rows, write_config.vector_dim,
                       write_config.string_length, write_config.needed_columns),
        st);
    batches.emplace_back(batch);
  }

  for (auto _ : st) {
    GBENCH_ASSERT_AND_ASSIGN(auto policy, ColumnGroupPolicy::create_column_group_policy(properties, schema), st);
    auto writer = Writer::create(base_path, schema, std::move(policy), std::move(properties));
    for (size_t i = 0; i < target_write_times; ++i) {
      auto rb = batches[i % multi_random_rb_counts];
      GBENCH_ASSERT_STATUS_OK(writer->write(rb), st);
      if ((i + 1) % multi_random_rb_counts == 0) {
        GBENCH_ASSERT_STATUS_OK(writer->flush(), st);
      }
    }

    GBENCH_ASSERT_AND_ASSIGN(auto cgs, writer->close(), st);
  }
}

BENCHMARK_DEFINE_F(StorageFixture, WriteDefaultConfig)(benchmark::State& st) {
  size_t loop_times = st.range(0);
  APIWriteBenchmark(st, fs_, base_path_, properties_, loop_times,
                    WriteDataConfig{.num_rows = 4096,
                                    .vector_dim = 128,
                                    .string_length = 128,
                                    .random_data = true,
                                    .needed_columns = {true, true, true, true}});
}

BENCHMARK_DEFINE_F(StorageFixture, WriteSingleColumnConfig)(benchmark::State& st) {
  size_t loop_times = st.range(0);
  int column_idx = st.range(1);

  std::array<bool, 4> needed_columns = {false, false, false, false};
  needed_columns[column_idx] = true;
  APIWriteBenchmark(st, fs_, base_path_, properties_, loop_times,
                    WriteDataConfig{.num_rows = 4096,
                                    .vector_dim = 128,
                                    .string_length = 128,
                                    .random_data = true,
                                    .needed_columns = needed_columns});
}

BENCHMARK_DEFINE_F(StorageFixture, ReadFullScanDefaultConfig)(benchmark::State& st) {
  size_t loop_times = st.range(0);

  APIFullReadBenchmark(st, fs_, base_path_, properties_, loop_times,
                       WriteDataConfig{.num_rows = 4096,
                                       .vector_dim = 128,
                                       .string_length = 128,
                                       .random_data = true,
                                       .needed_columns = {true, true, true, true}},
                       FullReadDataConfig{
                           .needed_columns = {true, true, true, true},
                       });
}

BENCHMARK_DEFINE_F(StorageFixture, ReadFullScanSingleColumnConfig)(benchmark::State& st) {
  size_t loop_times = st.range(0);
  int column_idx = st.range(1);

  std::array<bool, 4> needed_columns = {false, false, false, false};
  needed_columns[column_idx] = true;
  APIFullReadBenchmark(st, fs_, base_path_, properties_, loop_times,
                       WriteDataConfig{
                           .num_rows = 4096,
                           .vector_dim = 128,
                           .string_length = 128,
                           .random_data = true,
                           .needed_columns = {true, true, true, true},
                       },
                       FullReadDataConfig{
                           .needed_columns = needed_columns,
                       });
}

BENCHMARK_DEFINE_F(StorageFixture, WriteRead768dimVector)(benchmark::State& st) {
  uint64_t target_size = st.range(0);
  uint32_t target_dim = st.range(1);
  size_t num_rows = target_size / (target_dim * 8 /* float size*/);

  APIWriteLargeBenchmark(st, fs_, base_path_, properties_, num_rows, 30,
                         WriteDataConfig{.num_rows = 4096,
                                         .vector_dim = 768,
                                         .string_length = 128,
                                         .random_data = true,
                                         .needed_columns = {false, false, false, true}},
                         FullReadDataConfig{.needed_columns = {false, false, false, true}});
}

BENCHMARK_REGISTER_F(StorageFixture, WriteDefaultConfig)->Args({10});
BENCHMARK_REGISTER_F(StorageFixture, WriteSingleColumnConfig)->ArgsProduct({{10}, {0, 1, 2, 3}});
BENCHMARK_REGISTER_F(StorageFixture, ReadFullScanDefaultConfig)->Args({10});
BENCHMARK_REGISTER_F(StorageFixture, ReadFullScanSingleColumnConfig)->ArgsProduct({{10}, {0, 1, 2, 3}});
BENCHMARK_REGISTER_F(StorageFixture, WriteRead768dimVector)
    ->Args({6ULL * 1024 * 1024 * 1024 /* target file size */, 768 /* target dim */})
    ->Iterations(1);

}  // namespace milvus_storage