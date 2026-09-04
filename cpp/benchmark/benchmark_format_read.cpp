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
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include "iceberg_bridge.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/thread_pool.h"

namespace milvus_storage::benchmark {

using namespace milvus_storage::api;

namespace {

enum class ReaderBenchmarkMode { kSync, kAsync };
enum class ReaderBenchmarkOperation { kRecordBatchRead, kChunkRead, kTake };
enum class ReaderBenchmarkFormat { kParquet, kVortex, kLance };

struct ReaderBenchmarkConfig {
  ReaderBenchmarkMode mode;
  ReaderBenchmarkOperation operation;
  ReaderBenchmarkFormat format;
  ReaderBenchmarkDataset dataset;
};

static arrow::Result<std::string> ReaderBenchmarkSuffix(const ReaderBenchmarkConfig& config) {
  std::string_view mode_name;
  switch (config.mode) {
    case ReaderBenchmarkMode::kSync:
      mode_name = "Sync";
      break;
    case ReaderBenchmarkMode::kAsync:
      mode_name = "Async";
      break;
    default:
      return arrow::Status::Invalid("Unknown Reader benchmark mode");
  }

  std::string_view operation_name;
  switch (config.operation) {
    case ReaderBenchmarkOperation::kRecordBatchRead:
      operation_name = "RecordBatchRead";
      break;
    case ReaderBenchmarkOperation::kChunkRead:
      operation_name = "ChunkRead";
      break;
    case ReaderBenchmarkOperation::kTake:
      operation_name = "Take";
      break;
    default:
      return arrow::Status::Invalid("Unknown Reader benchmark operation");
  }

  std::string_view format_name;
  switch (config.format) {
    case ReaderBenchmarkFormat::kParquet:
      format_name = "Parquet";
      break;
    case ReaderBenchmarkFormat::kVortex:
      format_name = "Vortex";
      break;
    case ReaderBenchmarkFormat::kLance:
      format_name = "Lance";
      break;
    default:
      return arrow::Status::Invalid("Unknown Reader benchmark format");
  }

  return std::string(mode_name) + "/" + std::string(operation_name) + "/" + std::string(format_name) + "/" +
         std::string(ReaderBenchmarkDatasetName(config.dataset));
}

static arrow::Result<size_t> GetReaderBenchmarkExecutorThreads() {
  const auto* value = std::getenv("NIGHTLY_CI_EXECUTOR_THREADS");
  if (value == nullptr) {
    return size_t{1};
  }

  size_t threads = 0;
  const auto* end = value + std::strlen(value);
  const auto parse_result = std::from_chars(value, end, threads);
  if (value == end || parse_result.ec != std::errc{} || parse_result.ptr != end || threads == 0) {
    return arrow::Status::Invalid("NIGHTLY_CI_EXECUTOR_THREADS must be a positive integer");
  }
  return threads;
}

std::atomic<uint64_t> g_reader_benchmark_prefix_sequence{0};

struct ReaderBenchmarkMetrics {
  int64_t rows = 0;
  int64_t bytes = 0;
};

static arrow::Result<ReaderBenchmarkMetrics> RunRecordBatchReadOnce(
    Reader& reader, std::shared_ptr<arrow::RecordBatchReader>* batch_reader_out) {
  ARROW_ASSIGN_OR_RAISE(auto batch_reader, reader.get_record_batch_reader());
  *batch_reader_out = std::move(batch_reader);

  ReaderBenchmarkMetrics metrics;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    ARROW_RETURN_NOT_OK((*batch_reader_out)->ReadNext(&batch));
    if (!batch) {
      break;
    }
    metrics.rows += batch->num_rows();
    metrics.bytes += FormatBenchFixtureBase<>::CalculateRawDataSize(batch);
  }
  return metrics;
}

static arrow::Result<ReaderBenchmarkMetrics> RunTakeOnce(Reader& reader,
                                                         const std::vector<int64_t>& indices,
                                                         std::shared_ptr<arrow::Table>* table_out) {
  ARROW_ASSIGN_OR_RAISE(auto table, reader.take(indices));
  *table_out = std::move(table);

  ReaderBenchmarkMetrics metrics;
  metrics.rows = (*table_out)->num_rows();
  arrow::TableBatchReader batch_reader(**table_out);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    ARROW_RETURN_NOT_OK(batch_reader.ReadNext(&batch));
    if (!batch) {
      break;
    }
    metrics.bytes += FormatBenchFixtureBase<>::CalculateRawDataSize(batch);
  }
  return metrics;
}

class ReaderBenchmarkCase {
  public:
  static arrow::Result<std::unique_ptr<ReaderBenchmarkCase>> Make(const ReaderBenchmarkConfig& config) {
    auto benchmark_case = std::unique_ptr<ReaderBenchmarkCase>(new ReaderBenchmarkCase(config));
    ARROW_RETURN_NOT_OK(benchmark_case->Prepare());
    return benchmark_case;
  }

  arrow::Status Run(::benchmark::State& state) {
    int64_t total_rows_read = 0;
    int64_t total_bytes_read = 0;

    for (auto _ : state) {
      table_.reset();
      batches_.clear();
      chunk_reader_.reset();
      batch_reader_.reset();
      reader_.reset();

      reader_ = Reader::create(column_groups_, schema_, nullptr, properties_);
      if (!reader_) {
        return arrow::Status::Invalid("Failed to create Reader benchmark reader");
      }

      switch (config_.operation) {
        case ReaderBenchmarkOperation::kRecordBatchRead: {
          ARROW_ASSIGN_OR_RAISE(auto metrics, RunRecordBatchReadOnce(*reader_, &batch_reader_));
          total_rows_read += metrics.rows;
          total_bytes_read += metrics.bytes;
          break;
        }
        case ReaderBenchmarkOperation::kChunkRead: {
          if (config_.mode == ReaderBenchmarkMode::kAsync) {
            auto open_try = std::move(reader_->get_chunk_reader_async(0)).via(executor_.get()).getTry();
            if (open_try.hasException()) {
              return arrow::Status::IOError("Exception in Reader benchmark async ChunkRead open: ",
                                            open_try.exception().what().toStdString());
            }
            ARROW_ASSIGN_OR_RAISE(chunk_reader_, std::move(open_try).value());
          } else {
            ARROW_ASSIGN_OR_RAISE(chunk_reader_, reader_->get_chunk_reader(0));
          }
          std::vector<int64_t> indices(chunk_reader_->total_number_of_chunks());
          std::iota(indices.begin(), indices.end(), 0);
          if (config_.mode == ReaderBenchmarkMode::kAsync) {
            auto chunks_try = std::move(chunk_reader_->get_chunks_async(indices)).via(executor_.get()).getTry();
            if (chunks_try.hasException()) {
              return arrow::Status::IOError("Exception in Reader benchmark async ChunkRead chunks: ",
                                            chunks_try.exception().what().toStdString());
            }
            ARROW_ASSIGN_OR_RAISE(batches_, std::move(chunks_try).value());
          } else {
            ARROW_ASSIGN_OR_RAISE(batches_, chunk_reader_->get_chunks(indices));
          }
          for (const auto& batch : batches_) {
            if (!batch) {
              return arrow::Status::Invalid("Reader benchmark chunk reader returned a null batch");
            }
            total_rows_read += batch->num_rows();
            total_bytes_read += FormatBenchFixtureBase<>::CalculateRawDataSize(batch);
          }
          break;
        }
        case ReaderBenchmarkOperation::kTake: {
          if (config_.mode == ReaderBenchmarkMode::kAsync) {
            auto table_try = std::move(reader_->take_async(take_indices_)).via(executor_.get()).getTry();
            if (table_try.hasException()) {
              return arrow::Status::IOError("Exception in Reader benchmark async Take: ",
                                            table_try.exception().what().toStdString());
            }
            ARROW_ASSIGN_OR_RAISE(table_, std::move(table_try).value());
            total_rows_read += table_->num_rows();
            arrow::TableBatchReader table_batch_reader(*table_);
            std::shared_ptr<arrow::RecordBatch> batch;
            while (true) {
              ARROW_RETURN_NOT_OK(table_batch_reader.ReadNext(&batch));
              if (!batch) {
                break;
              }
              total_bytes_read += FormatBenchFixtureBase<>::CalculateRawDataSize(batch);
            }
          } else {
            ARROW_ASSIGN_OR_RAISE(auto metrics, RunTakeOnce(*reader_, take_indices_, &table_));
            total_rows_read += metrics.rows;
            total_bytes_read += metrics.bytes;
          }
          break;
        }
        default:
          return arrow::Status::Invalid("Unknown Reader benchmark operation");
      }
    }

    ReportThroughput(state, total_bytes_read, total_rows_read);
    if (config_.operation == ReaderBenchmarkOperation::kTake) {
      state.counters["rows_taken"] =
          ::benchmark::Counter(static_cast<double>(take_indices_.size()), ::benchmark::Counter::kDefaults);
    }
    if (config_.mode == ReaderBenchmarkMode::kAsync) {
      state.counters["executor_threads"] =
          ::benchmark::Counter(static_cast<double>(executor_threads_), ::benchmark::Counter::kDefaults);
    }
    ARROW_ASSIGN_OR_RAISE(auto suffix, ReaderBenchmarkSuffix(config_));
    state.SetLabel(suffix);
    return arrow::Status::OK();
  }

  arrow::Status Cleanup() {
    table_.reset();
    batches_.clear();
    chunk_reader_.reset();
    batch_reader_.reset();
    reader_.reset();
    column_groups_.reset();
    schema_.reset();
    executor_.reset();
    loader_.reset();

    if (cleaned_ || !fs_ || prefix_.empty()) {
      cleaned_ = true;
      return arrow::Status::OK();
    }
    auto status = DeleteTestDir(fs_, prefix_);
    if (status.ok()) {
      cleaned_ = true;
    }
    return status;
  }

  ~ReaderBenchmarkCase() {
    if (!cleaned_) {
      const auto status = Cleanup();
      if (!status.ok()) {
        std::cerr << "Reader benchmark cleanup failed: " << status.ToString() << '\n';
      }
    }
  }

  private:
  explicit ReaderBenchmarkCase(ReaderBenchmarkConfig config) : config_(config) {}

  arrow::Status Prepare() {
    ARROW_ASSIGN_OR_RAISE(auto suffix, ReaderBenchmarkSuffix(config_));
    std::string_view storage_format;
    switch (config_.format) {
      case ReaderBenchmarkFormat::kParquet:
        storage_format = LOON_FORMAT_PARQUET;
        break;
      case ReaderBenchmarkFormat::kVortex:
        storage_format = LOON_FORMAT_VORTEX;
        break;
      case ReaderBenchmarkFormat::kLance:
        storage_format = LOON_FORMAT_LANCE_TABLE;
        break;
      default:
        return arrow::Status::Invalid("Unknown Reader benchmark storage format");
    }
    if (config_.mode == ReaderBenchmarkMode::kAsync) {
      if (config_.operation == ReaderBenchmarkOperation::kRecordBatchRead ||
          config_.format == ReaderBenchmarkFormat::kLance) {
        return arrow::Status::Invalid("Unsupported async Reader benchmark case");
      }
      ARROW_ASSIGN_OR_RAISE(executor_threads_, GetReaderBenchmarkExecutorThreads());
      executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(executor_threads_);
    }

    ARROW_RETURN_NOT_OK(InitTestProperties(properties_));
    api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "32768");
    ARROW_ASSIGN_OR_RAISE(fs_, GetFileSystem(properties_));
    prefix_ = "format_benchmark/reader/" + suffix + "/" +
              std::to_string(g_reader_benchmark_prefix_sequence.fetch_add(1, std::memory_order_relaxed));

    ARROW_ASSIGN_OR_RAISE(loader_, CreateReaderBenchmarkDataLoader(config_.dataset));
    ARROW_RETURN_NOT_OK(loader_->Load());
    schema_ = loader_->GetSchema();
    if (!schema_) {
      return arrow::Status::Invalid("Reader benchmark data loader returned a null schema");
    }
    constexpr size_t kTakeRows = 1000;
    if (loader_->NumRows() < static_cast<int64_t>(kTakeRows)) {
      return arrow::Status::Invalid("Reader benchmark Take requires at least 1000 source rows");
    }
    take_indices_ = GenerateRandomIndices(1000, loader_->NumRows(), 42);
    const bool sorted = std::is_sorted(take_indices_.begin(), take_indices_.end());
    const bool unique = std::adjacent_find(take_indices_.begin(), take_indices_.end()) == take_indices_.end();
    const bool in_bounds = std::all_of(take_indices_.begin(), take_indices_.end(),
                                       [&](int64_t index) { return index >= 0 && index < loader_->NumRows(); });
    bool non_sequential = false;
    for (size_t i = 0; i < take_indices_.size(); ++i) {
      if (take_indices_[i] != static_cast<int64_t>(i)) {
        non_sequential = true;
        break;
      }
    }
    if (take_indices_.size() != kTakeRows || !sorted || !unique || !in_bounds || !non_sequential) {
      return arrow::Status::Invalid("Reader benchmark Take indices violated the fixed-seed random selection contract");
    }

    if (config_.format == ReaderBenchmarkFormat::kLance) {
      return PrepareLance();
    }

    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(std::string(storage_format), schema_));
    auto writer = Writer::create(prefix_, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create Reader benchmark writer");
    }
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, loader_->GetRecordBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }
    ARROW_ASSIGN_OR_RAISE(column_groups_, writer->close());
    if (!column_groups_ || column_groups_->empty()) {
      return arrow::Status::Invalid("Reader benchmark writer returned no column groups");
    }
    return arrow::Status::OK();
  }

  arrow::Status PrepareLance() {
    ArrowFileSystemConfig fs_config;
    ARROW_RETURN_NOT_OK(ArrowFileSystemConfig::create_file_system_config(properties_, fs_config));
    if (fs_config.storage_type == "remote") {
      api::SetValue(properties_, "extfs.reader_benchmark.storage_type", "remote");
      api::SetValue(properties_, "extfs.reader_benchmark.cloud_provider", fs_config.cloud_provider.c_str());
      api::SetValue(properties_, "extfs.reader_benchmark.address", fs_config.address.c_str());
      api::SetValue(properties_, "extfs.reader_benchmark.bucket_name", fs_config.bucket_name.c_str());
      api::SetValue(properties_, "extfs.reader_benchmark.region", fs_config.region.c_str());
      api::SetValue(properties_, "extfs.reader_benchmark.access_key_id", fs_config.access_key_id.c_str());
      api::SetValue(properties_, "extfs.reader_benchmark.access_key_value", fs_config.access_key_value.c_str());
      if (fs_config.use_ssl) {
        api::SetValue(properties_, "extfs.reader_benchmark.use_ssl", "true");
      }
      if (fs_config.use_iam) {
        api::SetValue(properties_, "extfs.reader_benchmark.use_iam", "true");
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto lance_uri, lance::BuildLanceBaseUri(fs_config, prefix_));
    ARROW_ASSIGN_OR_RAISE(auto storage_options, lance::ToWriterOptions(fs_config));
    ARROW_ASSIGN_OR_RAISE(auto batch_reader, loader_->GetRecordBatchReader());
    ArrowArrayStream stream{};
    ARROW_RETURN_NOT_OK(arrow::ExportRecordBatchReader(batch_reader, &stream));
    ARROW_ASSIGN_OR_RAISE(auto fragment_ids, lance::BlockingDataset::WriteDataset(lance_uri, &stream, storage_options));
    ARROW_ASSIGN_OR_RAISE(auto dataset,
                          lance::BlockingDataset::Open(lance_uri, fs_, lance::ToReaderOptions(fs_config)));
    if (fragment_ids.empty()) {
      return arrow::Status::Invalid("Lance Reader benchmark preparation returned no fragments");
    }

    auto column_group = std::make_shared<api::ColumnGroup>();
    column_group->format = LOON_FORMAT_LANCE_TABLE;
    column_group->columns.reserve(schema_->num_fields());
    for (const auto& field : schema_->fields()) {
      column_group->columns.emplace_back(field->name());
    }

    int64_t total_rows = 0;
    const auto milvus_lance_uri = lance::ToMilvusLanceUri(lance_uri, fs_config.address);
    for (const auto fragment_id : fragment_ids) {
      ARROW_ASSIGN_OR_RAISE(auto row_count, dataset->GetFragmentRowCount(fragment_id));
      if (row_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - total_rows)) {
        return arrow::Status::Invalid("Lance fragment row range exceeds int64_t");
      }
      // ColumnGroupFile ranges are physical offsets within each fragment;
      // Reader handles logical concatenation across the files.
      column_group->files.emplace_back(ColumnGroupFile{
          .path = lance::MakeLanceUri(milvus_lance_uri, fragment_id),
          .start_index = 0,
          .end_index = static_cast<int64_t>(row_count),
      });
      total_rows += static_cast<int64_t>(row_count);
    }
    if (total_rows != loader_->NumRows()) {
      return arrow::Status::Invalid("Lance fragment row count does not match Reader benchmark dataset");
    }
    column_groups_ = std::make_shared<ColumnGroups>();
    column_groups_->emplace_back(std::move(column_group));
    return arrow::Status::OK();
  }

  const ReaderBenchmarkConfig config_;
  Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::unique_ptr<BenchmarkDataLoader> loader_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<ColumnGroups> column_groups_;
  std::string prefix_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  size_t executor_threads_ = 0;
  bool cleaned_ = false;

  std::unique_ptr<Reader> reader_;
  std::shared_ptr<arrow::RecordBatchReader> batch_reader_;
  std::unique_ptr<ChunkReader> chunk_reader_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::shared_ptr<arrow::Table> table_;
  std::vector<int64_t> take_indices_;
};

static void RunReaderBenchmarkCase(::benchmark::State& state, ReaderBenchmarkConfig config) {
  auto case_result = ReaderBenchmarkCase::Make(config);
  if (!case_result.ok()) {
    const auto message = case_result.status().ToString();
    state.SkipWithError(message.c_str());
    return;
  }

  auto benchmark_case = std::move(case_result).ValueOrDie();
  const auto run_status = benchmark_case->Run(state);
  const auto cleanup_status = benchmark_case->Cleanup();
  if (!run_status.ok()) {
    const auto message = run_status.ToString();
    state.SkipWithError(message.c_str());
  } else if (!cleanup_status.ok()) {
    const auto message = cleanup_status.ToString();
    state.SkipWithError(message.c_str());
  }
}

[[maybe_unused]] const bool kReaderBenchmarksRegistered = [] {
  constexpr std::array<std::string_view, 2> prefixes = {
      "ReaderBenchmark/",
      "NIGHTLY_CI_TARGET/",
  };
  constexpr std::array<ReaderBenchmarkDataset, 7> datasets = {
      ReaderBenchmarkDataset::kSyntheticSmall,    ReaderBenchmarkDataset::kSyntheticMedium,
      ReaderBenchmarkDataset::kSyntheticLarge,    ReaderBenchmarkDataset::kScalarMedium,
      ReaderBenchmarkDataset::kRandomVector64MiB, ReaderBenchmarkDataset::kLowEntropyVector256MiB,
      ReaderBenchmarkDataset::kRandomVector2GiB,
  };
  constexpr std::array<ReaderBenchmarkFormat, 3> formats = {
      ReaderBenchmarkFormat::kParquet,
      ReaderBenchmarkFormat::kVortex,
      ReaderBenchmarkFormat::kLance,
  };
  constexpr std::array<ReaderBenchmarkOperation, 3> operations = {
      ReaderBenchmarkOperation::kRecordBatchRead,
      ReaderBenchmarkOperation::kChunkRead,
      ReaderBenchmarkOperation::kTake,
  };

  for (const auto dataset : datasets) {
    for (const auto format : formats) {
      for (const auto operation : operations) {
        const ReaderBenchmarkConfig config{
            .mode = ReaderBenchmarkMode::kSync,
            .operation = operation,
            .format = format,
            .dataset = dataset,
        };
        const auto suffix = ReaderBenchmarkSuffix(config).ValueOrDie();
        for (const auto prefix : prefixes) {
          ::benchmark::RegisterBenchmark(std::string(prefix) + suffix,
                                         [config](::benchmark::State& state) { RunReaderBenchmarkCase(state, config); })
              ->Unit(::benchmark::kMillisecond)
              ->UseRealTime();
        }
      }
    }
  }

  constexpr std::array<ReaderBenchmarkFormat, 2> async_formats = {
      ReaderBenchmarkFormat::kParquet,
      ReaderBenchmarkFormat::kVortex,
  };
  constexpr std::array<ReaderBenchmarkOperation, 2> async_operations = {
      ReaderBenchmarkOperation::kChunkRead,
      ReaderBenchmarkOperation::kTake,
  };
  for (const auto dataset : datasets) {
    for (const auto format : async_formats) {
      for (const auto operation : async_operations) {
        const ReaderBenchmarkConfig config{
            .mode = ReaderBenchmarkMode::kAsync,
            .operation = operation,
            .format = format,
            .dataset = dataset,
        };
        const auto suffix = ReaderBenchmarkSuffix(config).ValueOrDie();
        for (const auto prefix : prefixes) {
          ::benchmark::RegisterBenchmark(std::string(prefix) + suffix,
                                         [config](::benchmark::State& state) { RunReaderBenchmarkCase(state, config); })
              ->Unit(::benchmark::kMillisecond)
              ->UseRealTime();
        }
      }
    }
  }
  return true;
}();

struct PreparedReaderFile {
  ColumnGroupFile file;
  std::shared_ptr<arrow::Schema> read_schema;
  std::vector<std::string> needed_columns;
};

constexpr size_t kOpenManyParquetFooterPaddingBytes = 486 * 1024;
constexpr const char* kOpenManyParquetFooterPaddingKey = "open_many_format_readers_rss_footer_padding";

int64_t GetCurrentRSS() {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    return static_cast<int64_t>(info.resident_size);
  }
  return 0;
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  long total_pages = 0;
  long resident_pages = 0;
  statm >> total_pages >> resident_pages;
  if (!statm || resident_pages < 0) {
    return 0;
  }
  return static_cast<int64_t>(resident_pages) * static_cast<int64_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

const std::vector<std::string>& GetOpenManyFormatReaderFormats() {
  static const std::vector<std::string> formats = {
      LOON_FORMAT_PARQUET,
      LOON_FORMAT_VORTEX,
      LOON_FORMAT_LANCE_TABLE,
      LOON_FORMAT_ICEBERG_TABLE,
  };
  return formats;
}

std::string GetOpenManyFormatReaderFormatByIndex(size_t idx) {
  const auto& formats = GetOpenManyFormatReaderFormats();
  assert(idx < formats.size() && "OpenManyFormatReadersRSS format index out of range");
  return formats[idx];
}

arrow::Status EnsureOpenFileLimit(size_t reader_count) {
  struct rlimit limit {};
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    return arrow::Status::IOError("Failed to read RLIMIT_NOFILE: ", std::strerror(errno));
  }

  constexpr size_t kOpenFileMargin = 1024;
  if (reader_count > std::numeric_limits<size_t>::max() - kOpenFileMargin) {
    return arrow::Status::Invalid("OpenManyFormatReadersRSS reader_count is too large: ", reader_count);
  }

  auto required = static_cast<rlim_t>(reader_count + kOpenFileMargin);
  if (limit.rlim_cur >= required) {
    return arrow::Status::OK();
  }
  if (limit.rlim_max != RLIM_INFINITY && limit.rlim_max < required) {
    return arrow::Status::Invalid("RLIMIT_NOFILE hard limit is too low for OpenManyFormatReadersRSS. [required=",
                                  required, ", hard_limit=", limit.rlim_max, "]");
  }

  auto new_limit = limit;
  new_limit.rlim_cur = required;
  if (setrlimit(RLIMIT_NOFILE, &new_limit) != 0) {
    return arrow::Status::IOError("Failed to raise RLIMIT_NOFILE for OpenManyFormatReadersRSS. [required=", required,
                                  ", error=", std::strerror(errno), "]");
  }
  return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> ProjectSchema(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::vector<std::string>& columns) {
  if (!schema || columns.empty()) {
    return schema;
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (const auto& column : columns) {
    auto field = schema->GetFieldByName(column);
    if (field) {
      fields.emplace_back(std::move(field));
    }
  }
  return arrow::schema(std::move(fields));
}

template <typename ReaderT>
arrow::Result<std::vector<std::shared_ptr<FormatReader>>> OpenManyReadersFromSharedMetadata(
    const ColumnGroupFile& file,
    const Properties& properties,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    size_t reader_count,
    int64_t* rss_after_metadata,
    uint64_t* metadata_cache_size) {
  ARROW_ASSIGN_OR_RAISE(auto metadata, FormatReader::load_metadata<ReaderT>(file, properties, nullptr));
  if (metadata_cache_size) {
    *metadata_cache_size = metadata ? metadata->cache_size : 0;
  }
  if (rss_after_metadata) {
    *rss_after_metadata = GetCurrentRSS();
  }

  std::vector<std::shared_ptr<FormatReader>> readers;
  readers.reserve(reader_count);
  for (size_t i = 0; i < reader_count; ++i) {
    ARROW_ASSIGN_OR_RAISE(auto reader,
                          FormatReader::create_from_metadata<ReaderT>(metadata, file, read_schema, needed_columns, ""));
    readers.emplace_back(std::move(reader));
  }
  return readers;
}

arrow::Result<std::vector<std::shared_ptr<FormatReader>>> OpenManyReadersFromSharedMetadata(
    const std::string& format,
    const ColumnGroupFile& file,
    const Properties& properties,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    size_t reader_count,
    int64_t* rss_after_metadata,
    uint64_t* metadata_cache_size) {
  if (format == LOON_FORMAT_PARQUET) {
    return OpenManyReadersFromSharedMetadata<parquet::ParquetFormatReader>(
        file, properties, read_schema, needed_columns, reader_count, rss_after_metadata, metadata_cache_size);
  }
  if (format == LOON_FORMAT_VORTEX) {
    return OpenManyReadersFromSharedMetadata<vortex::VortexFormatReader>(
        file, properties, read_schema, needed_columns, reader_count, rss_after_metadata, metadata_cache_size);
  }
  if (format == LOON_FORMAT_LANCE_TABLE) {
    return OpenManyReadersFromSharedMetadata<lance::LanceTableReader>(
        file, properties, read_schema, needed_columns, reader_count, rss_after_metadata, metadata_cache_size);
  }
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    return OpenManyReadersFromSharedMetadata<iceberg::IcebergFormatReader>(
        file, properties, read_schema, needed_columns, reader_count, rss_after_metadata, metadata_cache_size);
  }
  return arrow::Status::Invalid("Unsupported OpenManyFormatReadersRSS format: ", format);
}

}  // namespace

//=============================================================================
// Read Performance Benchmark Base
//=============================================================================

class FormatReadBenchmark : public FormatBenchFixtureBase<> {
  public:
  void SetUp(::benchmark::State& st) override {
    FormatBenchFixtureBase<>::SetUp(st);

    // Get schema from data loader
    schema_ = GetLoaderSchema();
    BENCH_ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties_), st);
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
      if (!batch) {
        break;
      }
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

  arrow::Result<PreparedReaderFile> PrepareReaderFile(const std::string& format) {
    if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return PrepareIcebergReaderFile();
    }
    if (format == LOON_FORMAT_PARQUET) {
      return PreparePaddedParquetReaderFile();
    }

    auto path = GetUniquePath(format + "_open_many_reader_test");
    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));
    auto writer = Writer::create(path, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Failed to create writer for format: ", format);
    }

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, GetLoaderBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(writer->write(batch));
    }

    ARROW_ASSIGN_OR_RAISE(auto cgs, writer->close());
    if (!cgs) {
      return arrow::Status::Invalid("Writer returned null column groups for format: ", format);
    }

    for (const auto& column_group : *cgs) {
      if (!column_group || column_group->format != format || column_group->files.empty()) {
        continue;
      }
      return PreparedReaderFile{
          .file = column_group->files.front(),
          .read_schema = ProjectSchema(schema_, column_group->columns),
          .needed_columns = column_group->columns,
      };
    }
    return arrow::Status::Invalid("PrepareTestData did not produce a readable file for format: ", format);
  }

  arrow::Result<PreparedReaderFile> PreparePaddedParquetReaderFile() {
    auto path = GetUniquePath("parquet_open_many_reader_test") + "/data.parquet";
    ARROW_ASSIGN_OR_RAISE(auto writer, parquet::ParquetFileWriter::Make(fs_, schema_, path, properties_));

    ARROW_ASSIGN_OR_RAISE(auto batch_reader, GetLoaderBatchReader());
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      ARROW_RETURN_NOT_OK(writer->Write(batch));
    }

    std::string padding(kOpenManyParquetFooterPaddingBytes, 'x');
    ARROW_RETURN_NOT_OK(writer->AddUserMetadata({{kOpenManyParquetFooterPaddingKey, std::move(padding)}}));
    ARROW_ASSIGN_OR_RAISE(auto file, writer->Close());

    std::vector<std::string> needed_columns;
    needed_columns.reserve(schema_->num_fields());
    for (const auto& field : schema_->fields()) {
      needed_columns.emplace_back(field->name());
    }

    return PreparedReaderFile{
        .file = std::move(file),
        .read_schema = schema_,
        .needed_columns = std::move(needed_columns),
    };
  }

  arrow::Result<PreparedReaderFile> PrepareIcebergReaderFile() const {
    ARROW_ASSIGN_OR_RAISE(auto table_uri, MakeIcebergTableUri(GetUniquePath("iceberg_read_test")));

    ARROW_ASSIGN_OR_RAISE(auto storage_options, iceberg::ToWriterOptions(fs_config_));
    ARROW_ASSIGN_OR_RAISE(
        auto table_info,
        iceberg::CreateTestTable(table_uri, static_cast<uint64_t>(GetLoaderNumRows()), false, {}, storage_options));
    ARROW_ASSIGN_OR_RAISE(auto file_infos, iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, fs_,
                                                              iceberg::ToReaderOptions(fs_config_)));

    if (file_infos.empty()) {
      return arrow::Status::Invalid("Iceberg PlanFiles returned no files");
    }

    const auto& file_info = file_infos.front();
    auto data_file_path = file_info.data_file_path;
    if (fs_config_.storage_type == "local") {
      data_file_path = LocalReadablePath(data_file_path);
    } else {
      data_file_path = iceberg::ToMilvusUri(data_file_path, fs_config_.address);
    }

    std::unordered_map<std::string, std::string> file_properties;
    if (!file_info.delete_metadata_json.empty()) {
      file_properties[kPropertyMetadata] =
          fs_config_.storage_type == "local"
              ? std::string(file_info.delete_metadata_json.begin(), file_info.delete_metadata_json.end())
              : iceberg::ConvertDeleteMetadataPaths(file_info.delete_metadata_json, fs_config_.address);
    }

    return PreparedReaderFile{
        .file =
            ColumnGroupFile{
                .path = std::move(data_file_path),
                .start_index = 0,
                .end_index = static_cast<int64_t>(file_info.record_count),
                .properties = std::move(file_properties),
            },
        .read_schema = nullptr,
        .needed_columns = {"id", "name", "value"},
    };
  }

  arrow::Result<std::string> MakeIcebergTableUri(const std::string& relative_path) const {
    if (fs_config_.storage_type == "local") {
      return AbsoluteLocalPath(relative_path);
    }
    if (fs_config_.bucket_name.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set for remote Iceberg benchmark");
    }

    if (fs_config_.cloud_provider == kCloudProviderAzure) {
      return "abfss://" + fs_config_.bucket_name + "/" + relative_path;
    }
    if (fs_config_.cloud_provider == kCloudProviderGCP) {
      return "gs://" + fs_config_.bucket_name + "/" + relative_path;
    }
    if (fs_config_.cloud_provider == kCloudProviderAliyun) {
      return "oss://" + fs_config_.bucket_name + "/" + relative_path;
    }
    return "s3://" + fs_config_.bucket_name + "/" + relative_path;
  }

  std::string AbsoluteLocalPath(const std::string& relative_path) const {
    return (LocalRootPath() / relative_path).lexically_normal().string();
  }

  std::filesystem::path LocalRootPath() const {
    std::filesystem::path root(fs_config_.root_path);
    if (root.is_relative()) {
      root = std::filesystem::absolute(root);
    }
    std::error_code error;
    auto canonical_root = std::filesystem::weakly_canonical(root, error);
    return error ? root.lexically_normal() : canonical_root;
  }

  std::string LocalReadablePath(const std::string& path) const {
    auto root = LocalRootPath().string();
    std::filesystem::path local_path(path);
    std::error_code error;
    auto canonical_path = std::filesystem::weakly_canonical(local_path, error);
    auto normalized_path = (error ? local_path.lexically_normal() : canonical_path).string();
    auto prefix = root + "/";
    if (normalized_path.rfind(prefix, 0) == 0) {
      return normalized_path.substr(prefix.size());
    }
    return normalized_path;
  }

  std::shared_ptr<arrow::Schema> schema_;
  ArrowFileSystemConfig fs_config_;
};

//=============================================================================
// Full Scan Benchmark
//=============================================================================

// Measures public Reader record-batch full scans; data preparation stays outside the timed loop.
// Args: [format_idx, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadFullScan)(::benchmark::State& st) {
  auto format_idx = static_cast<size_t>(st.range(0));
  auto num_threads = static_cast<size_t>(st.range(1));
  auto memory_config_idx = static_cast<size_t>(st.range(2));

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

    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    BENCH_ASSERT_AND_ASSIGN(auto metrics, RunRecordBatchReadOnce(*reader, &batch_reader), st);
    total_rows_read += metrics.rows;
    total_bytes_read += metrics.bytes;
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

// Measures public Reader record-batch scans projected to a selected number of columns.
// Args: [format_idx, num_columns, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadProjection)(::benchmark::State& st) {
  auto format_idx = static_cast<size_t>(st.range(0));
  auto num_columns = static_cast<size_t>(st.range(1));
  auto num_threads = static_cast<size_t>(st.range(2));
  auto memory_config_idx = static_cast<size_t>(st.range(3));

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

// Measures public Reader::take for sequential, random, or clustered row-index distributions.
// Args: [format_idx, take_count, distribution, num_threads, memory_config_idx]
BENCHMARK_DEFINE_F(FormatReadBenchmark, ReadTake)(::benchmark::State& st) {
  auto format_idx = static_cast<size_t>(st.range(0));
  auto take_count = static_cast<size_t>(st.range(1));
  int distribution = static_cast<int>(st.range(2));
  auto num_threads = static_cast<size_t>(st.range(3));
  auto memory_config_idx = static_cast<size_t>(st.range(4));

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
  auto dist = static_cast<IndexDistribution>(distribution);
  auto indices = GenerateIndices(dist, take_count, GetLoaderNumRows());

  int64_t total_rows_read = 0;
  int64_t total_bytes_read = 0;

  for (auto _ : st) {
    auto reader = Reader::create(cgs, schema_, nullptr, properties_);
    BENCH_ASSERT_NOT_NULL(reader, st);

    std::shared_ptr<arrow::Table> table;
    BENCH_ASSERT_AND_ASSIGN(auto metrics, RunTakeOnce(*reader, indices, &table), st);
    total_rows_read += metrics.rows;
    total_bytes_read += metrics.bytes;
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
// FormatReader RSS Benchmark
//=============================================================================

// Open many live FormatReaders from one shared metadata payload and report RSS.
// Args: [format_idx, reader_count]
// Measures metadata and resident-memory cost while opening many readers for one prepared file.
BENCHMARK_DEFINE_F(FormatReadBenchmark, OpenManyFormatReadersRSS)(::benchmark::State& st) {
  auto format_idx = static_cast<size_t>(st.range(0));
  auto reader_count = static_cast<size_t>(st.range(1));

  const auto format = GetOpenManyFormatReaderFormatByIndex(format_idx);
  BENCH_ASSERT_STATUS_OK(EnsureOpenFileLimit(reader_count), st);

  BENCH_ASSERT_AND_ASSIGN(auto prepared, PrepareReaderFile(format), st);

  std::vector<std::shared_ptr<FormatReader>> readers;
  int64_t rss_before_metadata = 0;
  int64_t rss_after_metadata = 0;
  int64_t rss_after_readers = 0;
  uint64_t metadata_cache_size = 0;

  for (auto _ : st) {
    readers.clear();

    st.PauseTiming();
    rss_before_metadata = GetCurrentRSS();
    st.ResumeTiming();

    BENCH_ASSERT_AND_ASSIGN(readers,
                            OpenManyReadersFromSharedMetadata(format, prepared.file, properties_, prepared.read_schema,
                                                              prepared.needed_columns, reader_count,
                                                              &rss_after_metadata, &metadata_cache_size),
                            st);

    st.PauseTiming();
    rss_after_readers = GetCurrentRSS();
    st.ResumeTiming();
  }

  const auto metadata_rss_delta = std::max<int64_t>(0, rss_after_metadata - rss_before_metadata);
  const auto reader_rss_delta = std::max<int64_t>(0, rss_after_readers - rss_after_metadata);
  const auto total_rss_delta = std::max<int64_t>(0, rss_after_readers - rss_before_metadata);
  const auto reader_count_i64 = static_cast<int64_t>(reader_count);

  ReportSize(st, "rss_before_metadata", rss_before_metadata, ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_after_metadata", rss_after_metadata, ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_after_readers", rss_after_readers, ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_metadata_delta", metadata_rss_delta, ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_reader_delta", reader_rss_delta, ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_reader_delta_per_reader", reader_count_i64 > 0 ? reader_rss_delta / reader_count_i64 : 0,
             ::benchmark::Counter::kDefaults);
  ReportSize(st, "rss_total_delta", total_rss_delta, ::benchmark::Counter::kDefaults);
  ReportSize(st, "file_footer_size", static_cast<int64_t>(prepared.file.Get<uint64_t>(kPropertyFooterSize)),
             ::benchmark::Counter::kDefaults);
  ReportSize(st, "metadata_cache_size", static_cast<int64_t>(metadata_cache_size), ::benchmark::Counter::kDefaults);
  st.counters["reader_count"] =
      ::benchmark::Counter(static_cast<double>(reader_count), ::benchmark::Counter::kDefaults);
  st.SetLabel(format + "/" + std::to_string(reader_count) + "readers/" + GetDataDescription());
}

BENCHMARK_REGISTER_F(FormatReadBenchmark, OpenManyFormatReadersRSS)
    ->ArgsProduct({
        {0, 1, 2, 3},  // parquet, vortex, lance-table, iceberg-table
        {100, 10000},  // live FormatReader count
    })
    ->Iterations(1)
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace milvus_storage::benchmark
