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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/table.h>

#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::benchmark {

using namespace milvus_storage::api;

namespace {

enum class CrtIoMode { Block = 0, AsyncCrt = 1 };

const char* CrtIoModeName(CrtIoMode mode) {
  switch (mode) {
    case CrtIoMode::Block:
      return "block";
    case CrtIoMode::AsyncCrt:
      return "async-crt";
  }
  return "unknown";
}

struct ConcurrentReadMetrics {
  double open_wall_ms = 0;
  double warmup_wall_ms = 0;
  double read_wall_ms = 0;
  double wall_ms = 0;
  int peak_threads = 0;
  int64_t rows_read = 0;
  int64_t bytes_read = 0;
};

struct CrtDataConfig {
  size_t num_rows = 40960;
  size_t vector_dim = 128;
  size_t string_length = 128;
  size_t batch_rows = 40960;
  bool random_data = true;
  std::array<bool, 4> columns = {true, true, true, true};
  std::string label;
  std::string path_suffix;
};

struct PreparedData {
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<ColumnGroups> column_groups;
  std::shared_ptr<std::vector<std::string>> projection;
  uint64_t file_size = 0;
  size_t chunk_count = 0;
};

constexpr size_t kTakeRowsPerReader = 3;
constexpr size_t kTakeVectorDim = 256;
constexpr size_t kTakeVectorRowBytes = kTakeVectorDim * sizeof(float);
constexpr size_t kTakeRowStride = 2049;
constexpr size_t kTakeNumRows = 40960;
constexpr size_t kReadNumRows = 40960;
constexpr size_t kReadVectorDim = 128;
constexpr size_t kReadThreads = 512;
constexpr size_t kGetAllChunksThreads = 128;
constexpr size_t kGetAllChunksRawBytes = 64ULL * 1024 * 1024;
constexpr size_t kGetAllChunksNumRows = kGetAllChunksRawBytes / kTakeVectorRowBytes;
constexpr size_t kGetAllChunks2GiBThreads = 10;
constexpr size_t kGetAllChunks2GiBRawBytes = 2ULL * 1024 * 1024 * 1024;
constexpr size_t kGetAllChunks2GiBNumRows = kGetAllChunks2GiBRawBytes / kTakeVectorRowBytes;
constexpr size_t kGetAllChunksCompressedThreads = 128;
constexpr size_t kGetAllChunksCompressedRawBytes = 256ULL * 1024 * 1024;
constexpr size_t kGetAllChunksCompressedNumRows = kGetAllChunksCompressedRawBytes / kTakeVectorRowBytes;
constexpr size_t kCrtReaderBatchRawBytes = 64ULL * 1024 * 1024;
constexpr size_t kCrtReaderIoWarmupReads = 1;

CrtDataConfig FullSyntheticConfig() {
  return {
      .num_rows = kReadNumRows,
      .vector_dim = kReadVectorDim,
      .string_length = 128,
      .batch_rows = kReadNumRows,
      .random_data = true,
      .columns = {true, true, true, true},
      .label = "synthetic/" + std::to_string(kReadNumRows) + "rows/" + std::to_string(kReadVectorDim) + "dim",
      .path_suffix = "crt_reader_full",
  };
}

CrtDataConfig VectorConfig(size_t num_rows, bool random_data, std::string label, std::string path_suffix) {
  return {
      .num_rows = num_rows,
      .vector_dim = kTakeVectorDim,
      .string_length = 0,
      .batch_rows = std::max<size_t>(1, kCrtReaderBatchRawBytes / kTakeVectorRowBytes),
      .random_data = random_data,
      .columns = {false, false, false, true},
      .label = std::move(label),
      .path_suffix = std::move(path_suffix),
  };
}

std::shared_ptr<std::vector<std::string>> VectorProjection() {
  return std::make_shared<std::vector<std::string>>(std::initializer_list<std::string>{"vector"});
}

std::vector<int64_t> TakeIndicesForReader(size_t reader_id, size_t rows_per_take) {
  std::vector<int64_t> indices;
  indices.reserve(rows_per_take);
  for (size_t i = 0; i < rows_per_take; ++i) {
    indices.emplace_back(static_cast<int64_t>(reader_id + i * kTakeRowStride));
  }
  return indices;
}

double MillisSince(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}


void ReleaseUnusedArrowMemory() {
  arrow::default_memory_pool()->ReleaseUnused();
}

int64_t CalculateRecordBatchRawDataSize(const std::shared_ptr<arrow::RecordBatch>& batch) {
  int64_t size = 0;
  const auto num_rows = batch->num_rows();
  for (int i = 0; i < batch->num_columns(); ++i) {
    auto type = batch->column(i)->type();
    if (type->id() == arrow::Type::LIST) {
      auto list_array = std::static_pointer_cast<arrow::ListArray>(batch->column(i));
      const auto total_values = list_array->value_offset(num_rows) - list_array->value_offset(0);
      size += total_values * list_array->value_type()->byte_width();
    } else if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
      auto list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
      size += num_rows * list_type->list_size() * list_type->value_type()->byte_width();
    } else if (arrow::is_fixed_width(*type)) {
      size += num_rows * type->byte_width();
    } else if (type->id() == arrow::Type::STRING || type->id() == arrow::Type::BINARY) {
      auto offsets = batch->column(i)->data()->buffers[1];
      auto offset_base = batch->column(i)->offset();
      auto off_ptr = reinterpret_cast<const int32_t*>(offsets->data());
      size += off_ptr[offset_base + num_rows] - off_ptr[offset_base];
    } else {
      throw std::runtime_error("CalculateRecordBatchRawDataSize: unsupported type " + type->ToString());
    }
  }
  return size;
}

int64_t CalculateTableRawDataSize(const std::shared_ptr<arrow::Table>& table) {
  if (!table) {
    return 0;
  }
  arrow::TableBatchReader batch_reader(*table);
  int64_t bytes = 0;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (batch_reader.ReadNext(&batch).ok() && batch) {
    bytes += CalculateRecordBatchRawDataSize(batch);
  }
  return bytes;
}

class CrtSyntheticBatchReader final : public arrow::RecordBatchReader {
 public:
  CrtSyntheticBatchReader(std::shared_ptr<arrow::Schema> schema, CrtDataConfig config)
      : schema_(std::move(schema)), config_(std::move(config)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    if (rows_read_ >= config_.num_rows) {
      *out = nullptr;
      return arrow::Status::OK();
    }
    const auto rows = std::min(config_.batch_rows, config_.num_rows - rows_read_);
    ARROW_ASSIGN_OR_RAISE(*out,
                          CreateTestData(schema_,
                                         static_cast<int64_t>(rows_read_),
                                         config_.random_data,
                                         rows,
                                         config_.vector_dim,
                                         config_.string_length,
                                         config_.columns));
    rows_read_ += rows;
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::Schema> schema_;
  CrtDataConfig config_;
  size_t rows_read_ = 0;
};

class CrtDataLoader {
 public:
  explicit CrtDataLoader(CrtDataConfig config) : config_(std::move(config)) {}

  arrow::Status Load() {
    ARROW_ASSIGN_OR_RAISE(schema_, CreateTestSchema(config_.columns, config_.vector_dim));
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Schema> schema() const { return schema_; }
  const CrtDataConfig& config() const { return config_; }

  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> GetRecordBatchReader() const {
    if (!schema_) {
      return arrow::Status::Invalid("CRT benchmark data loader is not loaded");
    }
    return std::make_shared<CrtSyntheticBatchReader>(schema_, config_);
  }

 private:
  CrtDataConfig config_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace

class CrtReaderBenchmark : public FormatBenchFixtureBase<false, false> {
 public:
  void TearDown(::benchmark::State& st) override {
    FormatBenchFixtureBase<false, false>::TearDown(st);
  }

 protected:
  api::Properties PropertiesForMode(const std::string& format, CrtIoMode mode) const {
    auto properties = properties_;
    api::SetValue(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "32768");
    api::SetValue(properties, PROPERTY_READER_METADATA_CACHE_ENABLE, "true");
    api::SetValue(properties, PROPERTY_FS_S3_CRT_ASYNC_READ, mode == CrtIoMode::AsyncCrt ? "true" : "false");
    if (format == LOON_FORMAT_VORTEX) {
      api::SetValue(properties, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, "2");
    }
    return properties;
  }

  bool CheckCommonRequirements(::benchmark::State& st, const std::string& format, CrtIoMode mode) {
    if (!IsCloudEnv()) {
      st.SkipWithError("CRT reader benchmark requires STORAGE_TYPE=remote");
      return false;
    }
    if (!CheckFormatAvailable(st, format)) {
      return false;
    }
#ifndef WITH_CRT
    if (mode == CrtIoMode::AsyncCrt) {
      st.SkipWithError("Async CRT mode requires WITH_CRT=True build");
      return false;
    }
#endif
    return true;
  }

  arrow::Status PrepareData(const std::string& format,
                            const api::Properties& properties,
                            const CrtDataConfig& config,
                            const std::shared_ptr<std::vector<std::string>>& projection,
                            PreparedData* out) {
    CrtDataLoader loader(config);
    ARROW_RETURN_NOT_OK(loader.Load());

    auto base_path = GetUniquePath(config.path_suffix);
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, loader.schema()));
    auto writer = Writer::create(base_path, loader.schema(), std::move(policy), properties);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create CRT reader benchmark writer");
    }

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, loader.GetRecordBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
    if (!cgs || cgs->empty() || !(*cgs)[0] || (*cgs)[0]->files.empty()) {
      return arrow::Status::Invalid("Prepared CRT reader benchmark data has no column-group files");
    }

    auto reader = Reader::create(cgs, loader.schema(), projection, properties);
    if (!reader) {
      return arrow::Status::Invalid("Failed to create CRT reader benchmark reader");
    }
    ARROW_ASSIGN_OR_RAISE(auto chunk_reader, reader->get_chunk_reader(0, projection));
    const auto chunk_count = chunk_reader->total_number_of_chunks();
    if (chunk_count == 0) {
      return arrow::Status::Invalid("Prepared CRT reader benchmark data has no readable chunks");
    }

    const auto& file = (*cgs)[0]->files[0];
    out->schema = loader.schema();
    out->column_groups = std::move(cgs);
    out->projection = projection;
    out->file_size = file.Get<uint64_t>(api::kPropertyFileSize);
    out->chunk_count = chunk_count;
    return arrow::Status::OK();
  }

  arrow::Result<ConcurrentReadMetrics> RunConcurrentChunkReaders(
      const PreparedData& prepared,
      const api::Properties& properties,
      size_t reader_count,
      size_t chunks_per_reader,
      size_t warmup_reads) {
    ConcurrentReadMetrics metrics;
    ThreadTracker thread_tracker;

    std::mutex mutex;
    std::condition_variable cv;
    bool start_open = false;
    bool start_warmup = false;
    bool start_read = false;
    size_t opened = 0;
    size_t open_failed = 0;
    size_t warmed = 0;
    std::string first_error;
    std::atomic<int64_t> rows_read{0};
    std::atomic<int64_t> bytes_read{0};

    auto fail = [&](const std::string& message, bool during_open) {
      std::lock_guard<std::mutex> lock(mutex);
      if (first_error.empty()) {
        first_error = message;
      }
      if (during_open) {
        ++open_failed;
      }
      cv.notify_all();
    };


    std::vector<std::thread> workers;
    workers.reserve(reader_count);
    thread_tracker.Start(std::chrono::milliseconds(1));
    for (size_t reader_id = 0; reader_id < reader_count; ++reader_id) {
      workers.emplace_back([&, reader_id]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_open; });
        }

        auto reader = Reader::create(prepared.column_groups, prepared.schema, prepared.projection, properties);
        if (!reader) {
          fail("Failed to create top-level Reader", true);
          return;
        }
        auto maybe_chunk_reader = reader->get_chunk_reader(0, prepared.projection);
        if (!maybe_chunk_reader.ok()) {
          fail(maybe_chunk_reader.status().ToString(), true);
          return;
        }
        auto chunk_reader = std::move(maybe_chunk_reader).ValueOrDie();

        {
          std::lock_guard<std::mutex> lock(mutex);
          ++opened;
          cv.notify_all();
        }

        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_warmup || start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        for (size_t i = 0; i < warmup_reads; ++i) {
          const auto chunk_index = static_cast<int64_t>((reader_id + i) % prepared.chunk_count);
          auto maybe_batch = chunk_reader->get_chunk(chunk_index);
          if (!maybe_batch.ok()) {
            fail(maybe_batch.status().ToString(), false);
            return;
          }
        }

        if (warmup_reads > 0) {
          std::unique_lock<std::mutex> lock(mutex);
          ++warmed;
          cv.notify_all();
          cv.wait(lock, [&] { return start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        for (size_t i = 0; i < chunks_per_reader; ++i) {
          const auto chunk_index = static_cast<int64_t>((reader_id + i) % prepared.chunk_count);
          auto maybe_batch = chunk_reader->get_chunk(chunk_index);
          if (!maybe_batch.ok()) {
            fail(maybe_batch.status().ToString(), false);
            return;
          }
          auto batch = maybe_batch.ValueOrDie();
          if (batch) {
            rows_read.fetch_add(batch->num_rows(), std::memory_order_relaxed);
            bytes_read.fetch_add(CalculateRawDataSize(batch), std::memory_order_relaxed);
          }
        }
      });
    }

    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_open = true;
    }
    cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return opened + open_failed == reader_count; });
    }
    auto open_done = std::chrono::steady_clock::now();

    auto warmup_done = open_done;
    bool should_warmup = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      should_warmup = warmup_reads > 0 && first_error.empty();
    }
    if (should_warmup) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        start_warmup = true;
      }
      cv.notify_all();
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return warmed == opened || !first_error.empty(); });
      }
      warmup_done = std::chrono::steady_clock::now();
    }

    ReleaseUnusedArrowMemory();

    auto read_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_read = true;
    }
    cv.notify_all();

    for (auto& worker : workers) {
      worker.join();
    }
    auto done = std::chrono::steady_clock::now();
    thread_tracker.Stop();

    if (!first_error.empty()) {
      return arrow::Status::IOError(first_error);
    }

    metrics.open_wall_ms = MillisSince(start, open_done);
    metrics.warmup_wall_ms = MillisSince(open_done, warmup_done);
    metrics.read_wall_ms = MillisSince(read_start, done);
    metrics.wall_ms = MillisSince(start, done);
    metrics.peak_threads = thread_tracker.GetPeakThreads();
    metrics.rows_read = rows_read.load(std::memory_order_relaxed);
    metrics.bytes_read = bytes_read.load(std::memory_order_relaxed);
    return metrics;
  }

  arrow::Result<ConcurrentReadMetrics> RunConcurrentTakeReaders(const PreparedData& prepared,
                                                                const api::Properties& properties,
                                                                size_t reader_count,
                                                                size_t rows_per_take,
                                                                size_t warmup_reads) {
    ConcurrentReadMetrics metrics;
    ThreadTracker thread_tracker;

    std::mutex mutex;
    std::condition_variable cv;
    bool start_open = false;
    bool start_warmup = false;
    bool start_read = false;
    size_t opened = 0;
    size_t open_failed = 0;
    size_t warmed = 0;
    std::string first_error;
    std::atomic<int64_t> rows_read{0};
    std::atomic<int64_t> bytes_read{0};

    auto fail = [&](const std::string& message, bool during_open) {
      std::lock_guard<std::mutex> lock(mutex);
      if (first_error.empty()) {
        first_error = message;
      }
      if (during_open) {
        ++open_failed;
      }
      cv.notify_all();
    };


    std::vector<std::thread> workers;
    workers.reserve(reader_count);
    thread_tracker.Start(std::chrono::milliseconds(1));
    for (size_t reader_id = 0; reader_id < reader_count; ++reader_id) {
      workers.emplace_back([&, reader_id]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_open; });
        }

        auto reader = Reader::create(prepared.column_groups, prepared.schema, prepared.projection, properties);
        if (!reader) {
          fail("Failed to create top-level Reader", true);
          return;
        }
        auto metadata_probe = reader->get_chunk_reader(0, prepared.projection);
        if (!metadata_probe.ok()) {
          fail(metadata_probe.status().ToString(), true);
          return;
        }

        {
          std::lock_guard<std::mutex> lock(mutex);
          ++opened;
          cv.notify_all();
        }

        const auto indices = TakeIndicesForReader(reader_id, rows_per_take);

        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_warmup || start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        for (size_t i = 0; i < warmup_reads; ++i) {
          auto maybe_table = reader->take(indices, 1, prepared.projection);
          if (!maybe_table.ok()) {
            fail(maybe_table.status().ToString(), false);
            return;
          }
        }

        if (warmup_reads > 0) {
          std::unique_lock<std::mutex> lock(mutex);
          ++warmed;
          cv.notify_all();
          cv.wait(lock, [&] { return start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        auto maybe_table = reader->take(indices, 1, prepared.projection);
        if (!maybe_table.ok()) {
          fail(maybe_table.status().ToString(), false);
          return;
        }
        auto table = maybe_table.ValueOrDie();
        if (table) {
          rows_read.fetch_add(table->num_rows(), std::memory_order_relaxed);
          bytes_read.fetch_add(CalculateTableRawDataSize(table), std::memory_order_relaxed);
        }
      });
    }

    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_open = true;
    }
    cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return opened + open_failed == reader_count; });
    }
    auto open_done = std::chrono::steady_clock::now();

    auto warmup_done = open_done;
    bool should_warmup = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      should_warmup = warmup_reads > 0 && first_error.empty();
    }
    if (should_warmup) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        start_warmup = true;
      }
      cv.notify_all();
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return warmed == opened || !first_error.empty(); });
      }
      warmup_done = std::chrono::steady_clock::now();
    }

    ReleaseUnusedArrowMemory();

    auto read_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_read = true;
    }
    cv.notify_all();

    for (auto& worker : workers) {
      worker.join();
    }
    auto done = std::chrono::steady_clock::now();
    thread_tracker.Stop();

    if (!first_error.empty()) {
      return arrow::Status::IOError(first_error);
    }

    metrics.open_wall_ms = MillisSince(start, open_done);
    metrics.warmup_wall_ms = MillisSince(open_done, warmup_done);
    metrics.read_wall_ms = MillisSince(read_start, done);
    metrics.wall_ms = MillisSince(start, done);
    metrics.peak_threads = thread_tracker.GetPeakThreads();
    metrics.rows_read = rows_read.load(std::memory_order_relaxed);
    metrics.bytes_read = bytes_read.load(std::memory_order_relaxed);
    return metrics;
  }

  arrow::Result<ConcurrentReadMetrics> RunConcurrentGetAllChunksReaders(
      const PreparedData& prepared,
      const api::Properties& properties,
      const std::vector<int64_t>& chunk_indices,
      size_t reader_count,
      size_t warmup_reads) {
    ConcurrentReadMetrics metrics;
    ThreadTracker thread_tracker;

    std::mutex mutex;
    std::condition_variable cv;
    bool start_open = false;
    bool start_warmup = false;
    bool start_read = false;
    size_t opened = 0;
    size_t open_failed = 0;
    size_t warmed = 0;
    std::string first_error;
    std::atomic<int64_t> rows_read{0};
    std::atomic<int64_t> bytes_read{0};

    auto fail = [&](const std::string& message, bool during_open) {
      std::lock_guard<std::mutex> lock(mutex);
      if (first_error.empty()) {
        first_error = message;
      }
      if (during_open) {
        ++open_failed;
      }
      cv.notify_all();
    };


    std::vector<std::thread> workers;
    workers.reserve(reader_count);
    thread_tracker.Start(std::chrono::milliseconds(1));
    for (size_t reader_id = 0; reader_id < reader_count; ++reader_id) {
      workers.emplace_back([&]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_open; });
        }

        auto reader = Reader::create(prepared.column_groups, prepared.schema, prepared.projection, properties);
        if (!reader) {
          fail("Failed to create top-level Reader", true);
          return;
        }
        auto maybe_chunk_reader = reader->get_chunk_reader(0, prepared.projection);
        if (!maybe_chunk_reader.ok()) {
          fail(maybe_chunk_reader.status().ToString(), true);
          return;
        }
        auto chunk_reader = std::move(maybe_chunk_reader).ValueOrDie();

        {
          std::lock_guard<std::mutex> lock(mutex);
          ++opened;
          cv.notify_all();
        }

        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return start_warmup || start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        for (size_t i = 0; i < warmup_reads; ++i) {
          auto maybe_batches = chunk_reader->get_chunks(chunk_indices, 1);
          if (!maybe_batches.ok()) {
            fail(maybe_batches.status().ToString(), false);
            return;
          }
        }

        if (warmup_reads > 0) {
          std::unique_lock<std::mutex> lock(mutex);
          ++warmed;
          cv.notify_all();
          cv.wait(lock, [&] { return start_read || !first_error.empty(); });
          if (!first_error.empty()) {
            return;
          }
        }

        auto maybe_batches = chunk_reader->get_chunks(chunk_indices, 1);
        if (!maybe_batches.ok()) {
          fail(maybe_batches.status().ToString(), false);
          return;
        }
        for (const auto& batch : maybe_batches.ValueOrDie()) {
          if (batch) {
            rows_read.fetch_add(batch->num_rows(), std::memory_order_relaxed);
            bytes_read.fetch_add(CalculateRawDataSize(batch), std::memory_order_relaxed);
          }
        }
      });
    }

    auto start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_open = true;
    }
    cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return opened + open_failed == reader_count; });
    }
    auto open_done = std::chrono::steady_clock::now();

    auto warmup_done = open_done;
    bool should_warmup = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      should_warmup = warmup_reads > 0 && first_error.empty();
    }
    if (should_warmup) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        start_warmup = true;
      }
      cv.notify_all();
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return warmed == opened || !first_error.empty(); });
      }
      warmup_done = std::chrono::steady_clock::now();
    }

    ReleaseUnusedArrowMemory();

    auto read_start = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex);
      start_read = true;
    }
    cv.notify_all();

    for (auto& worker : workers) {
      worker.join();
    }
    auto done = std::chrono::steady_clock::now();
    thread_tracker.Stop();

    if (!first_error.empty()) {
      return arrow::Status::IOError(first_error);
    }

    metrics.open_wall_ms = MillisSince(start, open_done);
    metrics.warmup_wall_ms = MillisSince(open_done, warmup_done);
    metrics.read_wall_ms = MillisSince(read_start, done);
    metrics.wall_ms = MillisSince(start, done);
    metrics.peak_threads = thread_tracker.GetPeakThreads();
    metrics.rows_read = rows_read.load(std::memory_order_relaxed);
    metrics.bytes_read = bytes_read.load(std::memory_order_relaxed);
    return metrics;
  }

  void ReportCommon(::benchmark::State& st,
                    const PreparedData& prepared,
                    const CrtDataConfig& config,
                    const std::string& format,
                    CrtIoMode mode,
                    size_t reader_count,
                    size_t warmup_reads,
                    double total_open_wall_ms,
                    double total_warmup_wall_ms,
                    double total_read_wall_ms,
                    double total_wall_ms,
                    int peak_threads,
                    int64_t total_rows_read,
                    int64_t total_bytes_read,
                    const std::string& label_suffix) {
    st.counters["readers"] = ::benchmark::Counter(static_cast<double>(reader_count), ::benchmark::Counter::kDefaults);
    st.counters["warmup_reads"] =
        ::benchmark::Counter(static_cast<double>(warmup_reads), ::benchmark::Counter::kDefaults);
    st.counters["chunks"] = ::benchmark::Counter(static_cast<double>(prepared.chunk_count), ::benchmark::Counter::kDefaults);
    st.counters["open_wall_ms"] = ::benchmark::Counter(total_open_wall_ms, ::benchmark::Counter::kAvgIterations);
    st.counters["warmup_wall_ms"] = ::benchmark::Counter(total_warmup_wall_ms, ::benchmark::Counter::kAvgIterations);
    st.counters["read_wall_ms"] = ::benchmark::Counter(total_read_wall_ms, ::benchmark::Counter::kAvgIterations);
    st.counters["wall_ms"] = ::benchmark::Counter(total_wall_ms, ::benchmark::Counter::kAvgIterations);
    st.counters["peak_threads"] =
        ::benchmark::Counter(static_cast<double>(peak_threads), ::benchmark::Counter::kDefaults);
    ReportThroughput(st, total_bytes_read, total_rows_read);
    ReportSize(st, "read_payload", total_bytes_read);
    ReportSize(st, "file_size", static_cast<int64_t>(prepared.file_size));
    st.SetLabel(format + "/" + CrtIoModeName(mode) + "/" + std::to_string(reader_count) + "readers/" + label_suffix +
                "/" + config.label);
  }

  void RunGetAllChunksCase(::benchmark::State& st,
                           const std::string& format,
                           CrtIoMode mode,
                           size_t reader_count,
                           const CrtDataConfig& config,
                           size_t raw_bytes,
                           const std::string& label_suffix) {
    if (!CheckCommonRequirements(st, format, mode)) {
      return;
    }

    auto properties = PropertiesForMode(format, mode);

    PreparedData prepared;
    BENCH_ASSERT_STATUS_OK(PrepareData(format, properties, config, VectorProjection(), &prepared), st);

    std::vector<int64_t> chunk_indices;
    chunk_indices.reserve(prepared.chunk_count);
    for (size_t i = 0; i < prepared.chunk_count; ++i) {
      chunk_indices.emplace_back(static_cast<int64_t>(i));
    }

    double total_open_wall_ms = 0;
    double total_warmup_wall_ms = 0;
    double total_read_wall_ms = 0;
    double total_wall_ms = 0;
    int peak_threads = 0;
    int64_t total_rows_read = 0;
    int64_t total_bytes_read = 0;
    constexpr auto warmup_reads = kCrtReaderIoWarmupReads;

    for (auto _ : st) {
      BENCH_ASSERT_AND_ASSIGN(
          auto metrics,
          RunConcurrentGetAllChunksReaders(prepared, properties, chunk_indices, reader_count, warmup_reads),
          st);
      st.SetIterationTime(metrics.read_wall_ms / 1000.0);
      total_open_wall_ms += metrics.open_wall_ms;
      total_warmup_wall_ms += metrics.warmup_wall_ms;
      total_read_wall_ms += metrics.read_wall_ms;
      total_wall_ms += metrics.wall_ms;
      peak_threads = std::max(peak_threads, metrics.peak_threads);
      total_rows_read += metrics.rows_read;
      total_bytes_read += metrics.bytes_read;
    }

    const auto encoded_ratio =
        raw_bytes == 0 ? 0.0 : static_cast<double>(prepared.file_size) / static_cast<double>(raw_bytes);
    st.counters["encoded_ratio"] = ::benchmark::Counter(encoded_ratio, ::benchmark::Counter::kDefaults);
    ReportSize(st, "raw_file_payload", static_cast<int64_t>(raw_bytes));
    ReportCommon(st,
                 prepared,
                 config,
                 format,
                 mode,
                 reader_count,
                 warmup_reads,
                 total_open_wall_ms,
                 total_warmup_wall_ms,
                 total_read_wall_ms,
                 total_wall_ms,
                 peak_threads,
                 total_rows_read,
                 total_bytes_read,
                 label_suffix);
  }
};

BENCHMARK_DEFINE_F(CrtReaderBenchmark, ConcurrentOpenRead)(::benchmark::State& st) {
  const auto format = GetFormatByIndex(static_cast<size_t>(st.range(0)));
  const auto mode = static_cast<CrtIoMode>(st.range(1));
  const auto reader_count = static_cast<size_t>(st.range(2));
  constexpr size_t chunks_per_reader = 1;

  if (!CheckCommonRequirements(st, format, mode)) {
    return;
  }

  auto properties = PropertiesForMode(format, mode);
  PreparedData prepared;
  auto config = FullSyntheticConfig();
  BENCH_ASSERT_STATUS_OK(PrepareData(format, properties, config, nullptr, &prepared), st);

  double total_open_wall_ms = 0;
  double total_warmup_wall_ms = 0;
  double total_read_wall_ms = 0;
  double total_wall_ms = 0;
  int peak_threads = 0;
  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;
  constexpr auto warmup_reads = kCrtReaderIoWarmupReads;

  for (auto _ : st) {
    BENCH_ASSERT_AND_ASSIGN(auto metrics,
                            RunConcurrentChunkReaders(
                                prepared, properties, reader_count, chunks_per_reader, warmup_reads),
                            st);
    st.SetIterationTime(metrics.read_wall_ms / 1000.0);
    total_open_wall_ms += metrics.open_wall_ms;
    total_warmup_wall_ms += metrics.warmup_wall_ms;
    total_read_wall_ms += metrics.read_wall_ms;
    total_wall_ms += metrics.wall_ms;
    peak_threads = std::max(peak_threads, metrics.peak_threads);
    total_rows_read += metrics.rows_read;
    total_bytes_read += metrics.bytes_read;
  }

  st.counters["chunks_per_reader"] =
      ::benchmark::Counter(static_cast<double>(chunks_per_reader), ::benchmark::Counter::kDefaults);
  ReportCommon(st,
               prepared,
               config,
               format,
               mode,
               reader_count,
               warmup_reads,
               total_open_wall_ms,
               total_warmup_wall_ms,
               total_read_wall_ms,
               total_wall_ms,
               peak_threads,
               total_rows_read,
               total_bytes_read,
               std::to_string(chunks_per_reader) + "chunks");
}

BENCHMARK_REGISTER_F(CrtReaderBenchmark, ConcurrentOpenRead)
    ->Name("CrtReader/ConcurrentOpenRead")
    ->ArgsProduct({
        {0, 1},
        {static_cast<int64_t>(CrtIoMode::Block), static_cast<int64_t>(CrtIoMode::AsyncCrt)},
        {kReadThreads},
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_DEFINE_F(CrtReaderBenchmark, ConcurrentOpenTake)(::benchmark::State& st) {
  const auto format = GetFormatByIndex(static_cast<size_t>(st.range(0)));
  const auto mode = static_cast<CrtIoMode>(st.range(1));
  const auto reader_count = static_cast<size_t>(st.range(2));
  const auto rows_per_take = static_cast<size_t>(st.range(3));

  if (!CheckCommonRequirements(st, format, mode)) {
    return;
  }

  auto properties = PropertiesForMode(format, mode);

  PreparedData prepared;
  auto config = VectorConfig(kTakeNumRows,
                             false,
                             "vector/" + std::to_string(kTakeNumRows) + "rows/" +
                                 std::to_string(kTakeVectorDim) + "dim",
                             "crt_reader_take");
  BENCH_ASSERT_STATUS_OK(PrepareData(format, properties, config, VectorProjection(), &prepared), st);

  double total_open_wall_ms = 0;
  double total_warmup_wall_ms = 0;
  double total_read_wall_ms = 0;
  double total_wall_ms = 0;
  int peak_threads = 0;
  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;
  constexpr auto warmup_reads = kCrtReaderIoWarmupReads;

  for (auto _ : st) {
    BENCH_ASSERT_AND_ASSIGN(
        auto metrics,
        RunConcurrentTakeReaders(prepared, properties, reader_count, rows_per_take, warmup_reads),
        st);
    st.SetIterationTime(metrics.read_wall_ms / 1000.0);
    total_open_wall_ms += metrics.open_wall_ms;
    total_warmup_wall_ms += metrics.warmup_wall_ms;
    total_read_wall_ms += metrics.read_wall_ms;
    total_wall_ms += metrics.wall_ms;
    peak_threads = std::max(peak_threads, metrics.peak_threads);
    total_rows_read += metrics.rows_read;
    total_bytes_read += metrics.bytes_read;
  }

  st.counters["rows_per_take"] =
      ::benchmark::Counter(static_cast<double>(rows_per_take), ::benchmark::Counter::kDefaults);
  st.counters["take_row_stride"] =
      ::benchmark::Counter(static_cast<double>(kTakeRowStride), ::benchmark::Counter::kDefaults);
  ReportSize(st, "take_payload", total_bytes_read);
  ReportCommon(st,
               prepared,
               config,
               format,
               mode,
               reader_count,
               warmup_reads,
               total_open_wall_ms,
               total_warmup_wall_ms,
               total_read_wall_ms,
               total_wall_ms,
               peak_threads,
               total_rows_read,
               total_bytes_read,
               std::to_string(rows_per_take) + "rows_per_take");
}

BENCHMARK_REGISTER_F(CrtReaderBenchmark, ConcurrentOpenTake)
    ->Name("CrtReader/ConcurrentOpenTake")
    ->ArgsProduct({
        {0, 1},
        {static_cast<int64_t>(CrtIoMode::Block), static_cast<int64_t>(CrtIoMode::AsyncCrt)},
        {512},
        {kTakeRowsPerReader},
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_DEFINE_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunks)(::benchmark::State& st) {
  const auto format = GetFormatByIndex(static_cast<size_t>(st.range(0)));
  const auto mode = static_cast<CrtIoMode>(st.range(1));
  const auto reader_count = static_cast<size_t>(st.range(2));
  RunGetAllChunksCase(st,
                      format,
                      mode,
                      reader_count,
                      VectorConfig(kGetAllChunksNumRows,
                                   true,
                                   "vector/" + std::to_string(kGetAllChunksRawBytes / (1024 * 1024)) + "MiB_random",
                                   "crt_reader_get_all_chunks"),
                      kGetAllChunksRawBytes,
                      "get_all_chunks/64MiB_random_vector");
}

BENCHMARK_REGISTER_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunks)
    ->Name("CrtReader/ConcurrentOpenGetAllChunks")
    ->ArgsProduct({
        {0, 1},
        {static_cast<int64_t>(CrtIoMode::Block), static_cast<int64_t>(CrtIoMode::AsyncCrt)},
        {kGetAllChunksThreads},
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_DEFINE_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunks2GiB)(::benchmark::State& st) {
  const auto format = GetFormatByIndex(static_cast<size_t>(st.range(0)));
  const auto mode = static_cast<CrtIoMode>(st.range(1));
  const auto reader_count = static_cast<size_t>(st.range(2));
  RunGetAllChunksCase(st,
                      format,
                      mode,
                      reader_count,
                      VectorConfig(kGetAllChunks2GiBNumRows,
                                   true,
                                   "vector/2GiB_random",
                                   "crt_reader_get_all_chunks_2gib"),
                      kGetAllChunks2GiBRawBytes,
                      "get_all_chunks/2GiB_random_vector");
}

BENCHMARK_REGISTER_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunks2GiB)
    ->Name("CrtReader/ConcurrentOpenGetAllChunks2GiB")
    ->ArgsProduct({
        {0, 1},
        {static_cast<int64_t>(CrtIoMode::Block), static_cast<int64_t>(CrtIoMode::AsyncCrt)},
        {kGetAllChunks2GiBThreads},
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_DEFINE_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunksCompressed)(::benchmark::State& st) {
  const auto format = GetFormatByIndex(static_cast<size_t>(st.range(0)));
  const auto mode = static_cast<CrtIoMode>(st.range(1));
  const auto reader_count = static_cast<size_t>(st.range(2));
  RunGetAllChunksCase(st,
                      format,
                      mode,
                      reader_count,
                      VectorConfig(kGetAllChunksCompressedNumRows,
                                   false,
                                   "vector/256MiB_low_entropy",
                                   "crt_reader_get_all_chunks_compressed"),
                      kGetAllChunksCompressedRawBytes,
                      "get_all_chunks/256MiB_low_entropy_vector");
}

BENCHMARK_REGISTER_F(CrtReaderBenchmark, ConcurrentOpenGetAllChunksCompressed)
    ->Name("CrtReader/ConcurrentOpenGetAllChunksCompressed")
    ->ArgsProduct({
        {0, 1},
        {static_cast<int64_t>(CrtIoMode::Block), static_cast<int64_t>(CrtIoMode::AsyncCrt)},
        {kGetAllChunksCompressedThreads},
    })
    ->Unit(::benchmark::kMillisecond)
    ->UseManualTime();

}  // namespace milvus_storage::benchmark
