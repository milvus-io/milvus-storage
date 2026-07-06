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
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <arrow/table.h>
#include "iceberg_bridge.h"
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

    iceberg::IcebergTestTableInfo table_info;
    std::vector<iceberg::IcebergFileInfo> file_infos;
    try {
      auto storage_options = iceberg::ToStorageOptions(fs_config_);
      table_info =
          iceberg::CreateTestTable(table_uri, static_cast<uint64_t>(GetLoaderNumRows()), false, {}, storage_options);
      file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create Iceberg benchmark table: ", e.what());
    }

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

// Full scan benchmark - read all rows and all columns
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

// Column projection benchmark - read subset of columns
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

// Random access benchmark - read specific rows by indices
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

    BENCH_ASSERT_AND_ASSIGN(auto table, reader->take(indices), st);

    total_rows_read += table->num_rows();
    // Estimate bytes read
    arrow::TableBatchReader batch_reader(*table);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      BENCH_ASSERT_STATUS_OK(batch_reader.ReadNext(&batch), st);
      if (batch == nullptr) {
        break;
      }
      total_bytes_read += CalculateRawDataSize(batch);
    }
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

//=============================================================================
// Typical Benchmarks
// Run with: --benchmark_filter="Typical/"
//=============================================================================

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadFullScan)
    ->Name("Typical/ReadFullScan_Parquet")
    ->Args({0, 8, 1})  // Parquet + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadFullScan)
    ->Name("Typical/ReadFullScan_Vortex")
    ->Args({1, 8, 1})  // Vortex + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadProjection)
    ->Name("Typical/ReadProjection_Parquet")
    ->Args({0, 1, 8, 1})  // Parquet + 1 col + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadProjection)
    ->Name("Typical/ReadProjection_Vortex")
    ->Args({1, 1, 8, 1})  // Vortex + 1 col + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadTake)
    ->Name("Typical/ReadTake_Parquet")
    ->Args({0, 1000, 1, 8, 1})  // Parquet + 1000 rows + Random + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(FormatReadBenchmark, ReadTake)
    ->Name("Typical/ReadTake_Vortex")
    ->Args({1, 1000, 1, 8, 1})  // Vortex + 1000 rows + Random + 8 threads + Default
    ->Unit(::benchmark::kMillisecond)
    ->UseRealTime();

}  // namespace milvus_storage::benchmark
