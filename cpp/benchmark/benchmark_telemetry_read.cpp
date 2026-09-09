// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
// End-to-end tracing regression benchmark. Compile the same harness against
// both the pre-tracing library and current library; see the comparison runner.
#include <benchmark/benchmark.h>
#include <arrow/api.h>
#include <folly/init/Init.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/thread_pool.h"
#ifdef STORAGE_TELEMETRY_BENCHMARK
#include "milvus-storage/tracing.h"
#include <opentelemetry/exporters/memory/in_memory_span_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/samplers/parent.h>
#include <opentelemetry/sdk/trace/samplers/always_on.h>
#endif

namespace {
using namespace milvus_storage;
using namespace milvus_storage::api;
using Json = nlohmann::json;
constexpr int64_t kRows = 65536;
std::string root;
int mode = 0;
std::shared_ptr<arrow::Schema> Schema() {
  return arrow::schema({arrow::field("id", arrow::int64()), arrow::field("value", arrow::float64()),
                        arrow::field("vector", arrow::fixed_size_list(arrow::float32(), 32))});
}
void Check(const arrow::Status& status) {
  if (!status.ok())
    throw std::runtime_error(status.ToString());
}
template <typename T>
T Value(arrow::Result<T> result) {
  Check(result.status());
  return std::move(result).ValueOrDie();
}
Properties Props() {
  Properties p;
  for (const auto& [key, value] :
       std::vector<std::pair<const char*, std::string>>{{PROPERTY_FS_STORAGE_TYPE, "local"},
                                                        {PROPERTY_FS_ROOT_PATH, root},
                                                        {PROPERTY_READER_METADATA_CACHE_ENABLE, "true"},
                                                        {PROPERTY_READER_LOGICAL_CHUNK_ROWS, "4096"}}) {
    if (auto error = SetValue(p, key, value.c_str()))
      throw std::runtime_error(*error);
  }
  return p;
}
void Prepare() {
  auto p = Props();
  auto fs = Value(FilesystemCache::getInstance().get(p));
  arrow::Int64Builder ids;
  arrow::DoubleBuilder values;
  auto floats = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder vectors(arrow::default_memory_pool(), floats, 32);
  uint32_t rng = 1;
  for (int64_t i = 0; i < kRows; ++i) {
    Check(ids.Append(i));
    rng = rng * 1664525u + 1013904223u;
    Check(values.Append(static_cast<double>(rng) / 4294967296.0));
    Check(vectors.Append());
    for (int j = 0; j < 32; ++j) {
      rng = rng * 1664525u + 1013904223u;
      Check(floats->Append(static_cast<float>(rng) / 4294967296.0f));
    }
  }
  auto batch =
      arrow::RecordBatch::Make(Schema(), kRows, {Value(ids.Finish()), Value(values.Finish()), Value(vectors.Finish())});
  Json manifest;
  for (const std::string format : {"parquet", "vortex"}) {
    Check(fs->CreateDir(format));
    auto options = p;
    if (auto error = SetValue(options, PROPERTY_WRITER_POLICY, LOON_COLUMN_GROUP_POLICY_SINGLE))
      throw std::runtime_error(*error);
    if (auto error = SetValue(options, PROPERTY_WRITER_FORMAT, format.c_str()))
      throw std::runtime_error(*error);
    auto policy = Value(ColumnGroupPolicy::create_column_group_policy(options, Schema()));
    auto writer = Writer::create(format, Schema(), std::move(policy), options);
    Check(writer->write(batch));
    auto groups = Value(writer->close());
    auto& list = manifest[format] = Json::array();
    for (const auto& group : *groups) {
      Json item = {{"columns", group->columns}, {"format", group->format}, {"files", Json::array()}};
      for (const auto& file : group->files)
        item["files"].push_back({{"path", file.path},
                                 {"start", file.start_index},
                                 {"end", file.end_index},
                                 {"properties", file.properties}});
      list.push_back(std::move(item));
    }
  }
  std::ofstream(root + "/dataset.json") << manifest.dump(2);
}
std::shared_ptr<ColumnGroups> Groups(const std::string& format) {
  auto manifest = Json::parse(std::ifstream(root + "/dataset.json"));
  auto groups = std::make_shared<ColumnGroups>();
  for (const auto& item : manifest.at(format)) {
    auto group = std::make_shared<ColumnGroup>();
    group->columns = item.at("columns").get<std::vector<std::string>>();
    group->format = item.at("format").get<std::string>();
    for (const auto& file : item.at("files"))
      group->files.push_back({file.at("path").get<std::string>(), file.at("start").get<int64_t>(),
                              file.at("end").get<int64_t>(),
                              file.at("properties").get<std::unordered_map<std::string, std::string>>()});
    groups->push_back(std::move(group));
  }
  return groups;
}
template <typename F>
auto WithParent(F&& fn) {
#ifdef STORAGE_TELEMETRY_BENCHMARK
  if (mode != 0) {
    tracing::TraceParent parent;
    parent.trace_id[0] = 1;
    parent.span_id[0] = 2;
    parent.trace_flags = mode == 2 ? 0 : 1;
    auto scope = tracing::AttachParent(parent);
    return fn();
  }
#endif
  return fn();
}
// Metadata "cold" creates a fresh Reader/cache per operation, while file bytes
// stay in the OS page cache. It is not a cold disk/network measurement.
void Read(benchmark::State& state, const std::string& format, bool async, bool cold, int count) {
  try {
    auto groups = Groups(format);
    auto props = Props();
    auto reader = Reader::create(groups, Schema(), nullptr, props);
    folly::CPUThreadPoolExecutor executor(2);
    std::vector<int64_t> rows(count);
    for (int i = 0; i < count; ++i) rows[i] = (i * 7919) % kRows;
    std::sort(rows.begin(), rows.end());
    auto run = [&] {
      return WithParent([&] {
        if (cold)
          reader = Reader::create(groups, Schema(), nullptr, props);
        return async ? reader->take_async(rows, 2).via(&executor).get() : reader->take(rows, 2);
      });
    };
    // Both the OS page cache and runtime/thread pools are warm before timing.
    for (int i = 0; i < 3; ++i) {
      auto table = Value(run());
      if (table->num_rows() != count)
        throw std::runtime_error("incorrect take row count");
      auto ids = Value(table->CombineChunks())->GetColumnByName("id")->chunk(0);
      auto typed = std::static_pointer_cast<arrow::Int64Array>(ids);
      for (int j = 0; j < count; ++j)
        if (typed->Value(j) != rows[j])
          throw std::runtime_error("incorrect take order");
    }
    for (auto _ : state) {
      auto result = run();
      if (!result.ok()) {
        state.SkipWithError(result.status().ToString().c_str());
        break;
      }
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * count);
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}
void ScanOrChunk(benchmark::State& state, const std::string& format, bool chunks, bool async) {
  try {
    auto reader = Reader::create(Groups(format), Schema(), nullptr, Props());
    auto chunk_reader = Value(reader->get_chunk_reader(0));
    folly::CPUThreadPoolExecutor executor(2);
    auto run = [&] {
      return WithParent([&]() -> arrow::Result<int64_t> {
        if (chunks) {
          auto result =
              async ? chunk_reader->get_chunks_async({0}, 2).via(&executor).get() : chunk_reader->get_chunks({0}, 2);
          ARROW_ASSIGN_OR_RAISE(auto batches, std::move(result));
          int64_t count = 0;
          for (const auto& batch : batches) count += batch->num_rows();
          return count;
        }
        ARROW_ASSIGN_OR_RAISE(auto stream, reader->get_record_batch_reader());
        int64_t count = 0;
        while (true) {
          std::shared_ptr<arrow::RecordBatch> batch;
          ARROW_RETURN_NOT_OK(stream->ReadNext(&batch));
          if (!batch)
            break;
          count += batch->num_rows();
        }
        return count;
      });
    };
    const int64_t expected_rows = chunks ? Value(chunk_reader->get_chunk(0))->num_rows() : kRows;
    int64_t rows = 0;
    for (int i = 0; i < 3; ++i) {
      rows = Value(run());
      if (rows != expected_rows)
        throw std::runtime_error("incorrect scan/chunk row count");
    }
    for (auto _ : state) {
      auto result = run();
      if (!result.ok()) {
        state.SkipWithError(result.status().ToString().c_str());
        break;
      }
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * rows);
    state.counters["rows_per_operation"] = static_cast<double>(rows);
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}
void FileRead(benchmark::State& state, int bytes) {
  try {
    auto groups = Groups("parquet");
    auto fs = Value(FilesystemCache::getInstance().get(Props()));
    auto file = Value(fs->OpenInputFile(groups->at(0)->files.at(0).path));
    std::vector<uint8_t> buffer(bytes);
    for (auto _ : state) {
      auto result = WithParent([&] { return file->ReadAt(0, bytes, buffer.data()); });
      if (!result.ok()) {
        state.SkipWithError(result.status().ToString().c_str());
        break;
      }
      benchmark::DoNotOptimize(buffer.data());
    }
    state.SetBytesProcessed(state.iterations() * bytes);
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}
}  // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  folly::Init init(&argc, &argv, false);
  root = std::getenv("STORAGE_BENCH_DATA") ? std::getenv("STORAGE_BENCH_DATA") : "/tmp/storage-telemetry-bench-data";
  mode = std::getenv("STORAGE_TELEMETRY_MODE") ? std::atoi(std::getenv("STORAGE_TELEMETRY_MODE")) : 0;
  Check(ConfigureStorageRuntime(2, 2));
  ThreadPoolHolder::WithSingleton(2);
  if (std::getenv("STORAGE_BENCH_PREPARE")) {
    Prepare();
    return 0;
  }
#ifdef STORAGE_TELEMETRY_BENCHMARK
  if (mode >= 2) {
    namespace sdk = opentelemetry::sdk::trace;
    auto exporter = std::make_unique<opentelemetry::exporter::memory::InMemorySpanExporter>(1);
    auto processor = std::make_unique<sdk::SimpleSpanProcessor>(std::move(exporter));
    auto sampler = std::make_unique<sdk::ParentBasedSampler>(std::make_unique<sdk::AlwaysOnSampler>());
    tracing::SetTracerProvider(tracing::ProviderPtr(new sdk::TracerProvider(
        std::move(processor), opentelemetry::sdk::resource::Resource::Create({}), std::move(sampler))));
  }
#endif
  for (const std::string format : {"parquet", "vortex"})
    for (bool async : {false, true})
      for (bool cold : {false, true})
        for (int rows : {64, 1024}) {
          auto name = format + "/take_" + (async ? "async/" : "sync/") + (cold ? "cold_meta/" : "warm_meta/") +
                      std::to_string(rows);
          benchmark::RegisterBenchmark(name.c_str(), Read, format, async, cold, rows)
              ->UseRealTime()
              ->MeasureProcessCPUTime();
        }
  for (const std::string format : {"parquet", "vortex"}) {
    benchmark::RegisterBenchmark((format + "/full_scan").c_str(), ScanOrChunk, format, false, false)
        ->UseRealTime()
        ->MeasureProcessCPUTime();
    for (bool async : {false, true})
      benchmark::RegisterBenchmark((format + (async ? "/chunk_async" : "/chunk_sync")).c_str(), ScanOrChunk, format,
                                   true, async)
          ->UseRealTime()
          ->MeasureProcessCPUTime();
  }
  for (int bytes : {64, 4096, 65536})
    benchmark::RegisterBenchmark(("file/read_at/" + std::to_string(bytes)).c_str(), FileRead, bytes)
        ->UseRealTime()
        ->MeasureProcessCPUTime();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
#ifdef STORAGE_TELEMETRY_BENCHMARK
  tracing::SetTracerProvider(nullptr);
#endif
  ThreadPoolHolder::Release();
}
