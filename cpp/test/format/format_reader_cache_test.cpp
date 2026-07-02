// Copyright 2023 Zilliz
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

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arrow/api.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/format_reader_cache.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/thread_pool.h"
#include "milvus-storage/writer.h"
#include "iceberg_bridge.h"
#include "test_env.h"

namespace milvus_storage::test {
namespace {

using TestMetadataCaches = FormatReaderMetadataCaches<parquet::ParquetFormatReader,
                                                      vortex::VortexFormatReader,
                                                      lance::LanceTableReader,
                                                      iceberg::IcebergFormatReader>;

constexpr uint64_t kDefaultTestMetadataCacheCapacity = 256 * 1024 * 1024;

template <typename CacheT>
typename CacheT::MetadataPtr MakeMetadata(std::string cache_key, uint64_t cache_size = 1) {
  using Metadata = typename CacheT::Trait::Metadata;

  auto metadata = std::make_shared<Metadata>();
  metadata->cache_key = cache_key;
  metadata->path = "/tmp/" + cache_key;
  metadata->cache_size = cache_size;
  typename CacheT::MetadataPtr ptr = metadata;
  return ptr;
}

template <typename Visitor>
arrow::Status VisitFormatReaderMetadataCachesForFormat(const std::string& format,
                                                       const TestMetadataCaches& caches,
                                                       Visitor&& visitor) {
  if (format == LOON_FORMAT_PARQUET) {
    return std::forward<Visitor>(visitor)(caches.get<parquet::ParquetFormatReader>());
  }
  if (format == LOON_FORMAT_VORTEX) {
    return std::forward<Visitor>(visitor)(caches.get<vortex::VortexFormatReader>());
  }
  if (format == LOON_FORMAT_LANCE_TABLE) {
    return std::forward<Visitor>(visitor)(caches.get<lance::LanceTableReader>());
  }
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    return std::forward<Visitor>(visitor)(caches.get<iceberg::IcebergFormatReader>());
  }
  return arrow::Status::Invalid("Unknown column group format: ", format);
}

template <typename Visitor>
arrow::Status VisitGlobalMetadataCacheForFormat(const std::string& format, Visitor&& visitor) {
  if (format == LOON_FORMAT_PARQUET) {
    return std::forward<Visitor>(visitor)(GetGlobalFormatReaderMetadataCache<parquet::ParquetFormatReader>());
  }
  if (format == LOON_FORMAT_VORTEX) {
    return std::forward<Visitor>(visitor)(GetGlobalFormatReaderMetadataCache<vortex::VortexFormatReader>());
  }
  if (format == LOON_FORMAT_LANCE_TABLE) {
    return std::forward<Visitor>(visitor)(GetGlobalFormatReaderMetadataCache<lance::LanceTableReader>());
  }
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    return std::forward<Visitor>(visitor)(GetGlobalFormatReaderMetadataCache<iceberg::IcebergFormatReader>());
  }
  return arrow::Status::Invalid("Unknown column group format: ", format);
}

arrow::Status ValidateChunkRead(const std::vector<std::shared_ptr<arrow::RecordBatch>>& chunks,
                                const std::vector<std::string>& expected_columns,
                                int64_t expected_rows) {
  int64_t actual_rows = 0;
  for (const auto& chunk : chunks) {
    if (!chunk) {
      return arrow::Status::Invalid("chunk reader returned a null record batch");
    }
    if (chunk->num_columns() != static_cast<int>(expected_columns.size())) {
      return arrow::Status::Invalid("unexpected column count: ", chunk->num_columns(), " != ", expected_columns.size());
    }
    for (int column_index = 0; column_index < chunk->num_columns(); ++column_index) {
      const auto& expected_name = expected_columns[column_index];
      const auto& actual_name = chunk->schema()->field(column_index)->name();
      if (actual_name != expected_name) {
        return arrow::Status::Invalid("unexpected column name at index ", column_index, ": ", actual_name,
                                      " != ", expected_name);
      }
    }
    actual_rows += chunk->num_rows();
  }

  if (actual_rows != expected_rows) {
    return arrow::Status::Invalid("unexpected row count: ", actual_rows, " != ", expected_rows);
  }
  return arrow::Status::OK();
}

template <typename CacheT>
arrow::Status ExpectGlobalMetadataCacheEntries(const std::shared_ptr<api::ColumnGroups>& column_groups,
                                               const std::shared_ptr<CacheT>& cache,
                                               const std::string& format,
                                               const api::Properties& properties,
                                               bool expect_cached) {
  using ReaderT = typename CacheT::ReaderType;

  if (!column_groups) {
    return arrow::Status::Invalid("column groups is null");
  }

  size_t checked_files = 0;
  for (const auto& column_group : *column_groups) {
    if (!column_group || column_group->format != format) {
      continue;
    }
    for (const auto& file : column_group->files) {
      ARROW_ASSIGN_OR_RAISE(auto key,
                            MakeFormatReaderMetadataCacheKey(file, properties, ReaderT::MetaTrait::cache_key(file)));
      auto cached = cache->get(key);
      if (expect_cached) {
        EXPECT_TRUE(cached.has_value()) << "Expected cached metadata for key: " << key;
      } else {
        EXPECT_FALSE(cached.has_value()) << "Unexpected cached metadata for key: " << key;
      }
      ++checked_files;
    }
  }

  if (checked_files == 0) {
    return arrow::Status::Invalid("no files checked for format: ", format);
  }
  return arrow::Status::OK();
}

arrow::Status ExpectGlobalMetadataCacheEntriesForFormat(const std::shared_ptr<api::ColumnGroups>& column_groups,
                                                        const std::string& format,
                                                        const api::Properties& properties,
                                                        bool expect_cached) {
  return VisitGlobalMetadataCacheForFormat(format, [&](const auto& cache) -> arrow::Status {
    return ExpectGlobalMetadataCacheEntries(column_groups, cache, format, properties, expect_cached);
  });
}

arrow::Status ReadProjectedChunks(api::Reader* reader,
                                  const std::vector<std::string>& projected_columns,
                                  int64_t expected_rows,
                                  size_t parallelism) {
  auto needed_columns = std::make_shared<std::vector<std::string>>(projected_columns);
  ARROW_ASSIGN_OR_RAISE(auto chunk_reader, reader->get_chunk_reader(0, needed_columns));

  const auto total_chunks = chunk_reader->total_number_of_chunks();
  if (total_chunks == 0) {
    return arrow::Status::Invalid("chunk reader unexpectedly has no chunks");
  }

  std::vector<int64_t> chunk_indices(total_chunks);
  std::iota(chunk_indices.begin(), chunk_indices.end(), 0);

  ARROW_ASSIGN_OR_RAISE(auto chunks, chunk_reader->get_chunks(chunk_indices, parallelism));
  if (chunks.size() != chunk_indices.size()) {
    return arrow::Status::Invalid("unexpected chunk count: ", chunks.size(), " != ", chunk_indices.size());
  }

  return ValidateChunkRead(chunks, projected_columns, expected_rows);
}

std::vector<std::string> StressProjection(int thread_index, int64_t iteration) {
  switch ((thread_index + iteration) % 3) {
    case 0:
      return {"id"};
    case 1:
      return {"value"};
    default:
      return {"id", "name", "value"};
  }
}

template <typename CacheT>
arrow::Status RunThrowingLoaderNotifiesWaitersAndAllowsRetry(const std::shared_ptr<CacheT>& cache) {
  using MetadataPtr = typename CacheT::MetadataPtr;
  const std::string key = "throwing-loader-retry";
  auto metadata = MakeMetadata<CacheT>(key);

  std::mutex mutex;
  std::condition_variable cv;
  bool loader_entered = false;
  bool waiter_started = false;
  bool release_loader = false;

  std::promise<arrow::Result<MetadataPtr>> first_promise;
  auto first_call = first_promise.get_future();
  std::thread first_thread([&, promise = std::move(first_promise)]() mutable {
    promise.set_value(cache->get_or_open(key, [&]() -> arrow::Result<MetadataPtr> {
      std::unique_lock<std::mutex> lock(mutex);
      loader_entered = true;
      cv.notify_all();
      cv.wait(lock, [&]() { return release_loader; });
      throw std::runtime_error("metadata loader exploded");
    }));
  });

  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(5), [&]() { return loader_entered; })) {
      release_loader = true;
      cv.notify_all();
      first_thread.join();
      return arrow::Status::Invalid("timed out waiting for first metadata loader");
    }
  }

  std::promise<arrow::Result<MetadataPtr>> waiter_promise;
  auto waiter_call = waiter_promise.get_future();
  std::thread waiter_thread([&, promise = std::move(waiter_promise)]() mutable {
    {
      std::lock_guard<std::mutex> lock(mutex);
      waiter_started = true;
    }
    cv.notify_all();
    promise.set_value(cache->get_or_open(
        key, [&]() -> arrow::Result<MetadataPtr> { return arrow::Status::Invalid("waiter must not run loader"); }));
  });

  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(5), [&]() { return waiter_started; })) {
      release_loader = true;
      cv.notify_all();
      first_thread.join();
      waiter_thread.join();
      return arrow::Status::Invalid("timed out waiting for metadata cache waiter");
    }
    release_loader = true;
  }
  cv.notify_all();

  first_thread.join();
  waiter_thread.join();

  auto first_result = first_call.get();
  auto waiter_result = waiter_call.get();
  if (first_result.ok()) {
    return arrow::Status::Invalid("throwing metadata loader unexpectedly succeeded");
  }
  if (waiter_result.ok()) {
    return arrow::Status::Invalid("waiter unexpectedly succeeded after throwing metadata loader");
  }
  if (cache->get(key).has_value()) {
    return arrow::Status::Invalid("throwing metadata loader poisoned cache entry");
  }

  ARROW_ASSIGN_OR_RAISE(auto retried,
                        cache->get_or_open(key, [&]() -> arrow::Result<MetadataPtr> { return metadata; }));
  if (retried.get() != metadata.get()) {
    return arrow::Status::Invalid("retried metadata pointer mismatch");
  }
  return arrow::Status::OK();
}

template <typename CacheT>
arrow::Status RunSingleflightsConcurrentSameKey(const std::shared_ptr<CacheT>& cache) {
  using MetadataPtr = typename CacheT::MetadataPtr;
  const std::string key = "concurrent-singleflight";
  auto metadata = MakeMetadata<CacheT>(key);
  constexpr int kThreadCount = 8;

  std::atomic<int> loader_calls{0};
  std::atomic<int> calls_started{0};
  int threads_ready = 0;
  bool start = false;
  bool loader_entered = false;
  bool release_loader = false;
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<arrow::Status> statuses(kThreadCount);
  std::vector<MetadataPtr> results(kThreadCount);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++threads_ready;
        cv.notify_all();
        cv.wait(lock, [&]() { return start; });
      }

      ++calls_started;
      cv.notify_all();

      auto result = cache->get_or_open(key, [&]() -> arrow::Result<MetadataPtr> {
        ++loader_calls;
        std::unique_lock<std::mutex> lock(mutex);
        loader_entered = true;
        cv.notify_all();
        cv.wait(lock, [&]() { return release_loader; });
        return metadata;
      });

      statuses[i] = result.status();
      if (result.ok()) {
        results[i] = result.ValueOrDie();
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&]() { return threads_ready == kThreadCount; });
    start = true;
    cv.notify_all();
    if (!cv.wait_for(lock, std::chrono::seconds(5),
                     [&]() { return loader_entered && calls_started.load() == kThreadCount; })) {
      release_loader = true;
      cv.notify_all();
      lock.unlock();
      for (auto& thread : threads) {
        thread.join();
      }
      return arrow::Status::Invalid("timed out waiting for concurrent metadata cache callers");
    }
    release_loader = true;
  }
  cv.notify_all();

  for (auto& thread : threads) {
    thread.join();
  }

  if (loader_calls.load() != 1) {
    return arrow::Status::Invalid("expected one metadata loader call, got ", loader_calls.load());
  }
  for (int i = 0; i < kThreadCount; ++i) {
    if (!statuses[i].ok()) {
      return arrow::Status::Invalid("metadata cache caller ", i, " failed: ", statuses[i].ToString());
    }
    if (results[i].get() != metadata.get()) {
      return arrow::Status::Invalid("metadata cache caller ", i, " got unexpected metadata pointer");
    }
  }
  return arrow::Status::OK();
}

}  // namespace

class FormatReaderMetadataCacheParamTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    InitMetadataCache(kDefaultTestMetadataCacheCapacity);
    ClearMetadataCache();
  }

  void TearDown() override {
    InitMetadataCache(kDefaultTestMetadataCacheCapacity);
    ClearMetadataCache();
  }
};

class FormatReaderMetadataCacheStressTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    InitMetadataCache(kDefaultTestMetadataCacheCapacity);
    ClearMetadataCache();

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
    ASSERT_STATUS_OK(MirrorDefaultExternalFilesystemConfig());

    base_path_ = GetTestBasePath("format-reader-cache-stress-" + GetParam());
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));

    api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "true");
    api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "32");
    api::SetValue(properties_, PROPERTY_WRITER_FILE_ROLLING_SIZE, "1");

    ThreadPoolHolder::WithSingleton(kReadParallelism);
  }

  void TearDown() override {
    auto delete_status = DeleteTestDir(fs_, base_path_);
    ThreadPoolHolder::Release();
    InitMetadataCache(kDefaultTestMetadataCacheCapacity);
    ClearMetadataCache();
    ASSERT_STATUS_OK(delete_status);
  }

  arrow::Result<std::shared_ptr<api::ColumnGroups>> WriteStressData(const std::string& format) {
    ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));
    schema_ = schema;

    if (format == LOON_FORMAT_ICEBERG_TABLE) {
      return WriteIcebergStressData();
    }

    ARROW_ASSIGN_OR_RAISE(auto policy, CreateSinglePolicy(format, schema_));
    auto writer = api::Writer::create(base_path_, schema_, std::move(policy), properties_);
    if (!writer) {
      return arrow::Status::Invalid("Writer::create returned null");
    }

    for (int batch_index = 0; batch_index < kBatchCount; ++batch_index) {
      ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema_, batch_index * kRowsPerBatch, false, kRowsPerBatch));
      ARROW_RETURN_NOT_OK(writer->write(batch));
      if (batch_index + 1 < kBatchCount) {
        ARROW_RETURN_NOT_OK(writer->flush());
      }
    }

    return writer->close();
  }

  arrow::Status MirrorDefaultExternalFilesystemConfig() {
    if (fs_config_.storage_type != "remote") {
      return arrow::Status::OK();
    }

    auto set_extfs = [this](const std::string& key, const std::string& value) -> arrow::Status {
      auto error = api::SetValue(properties_, ("extfs.default." + key).c_str(), value.c_str());
      if (error) {
        return arrow::Status::Invalid(*error);
      }
      return arrow::Status::OK();
    };

    ARROW_RETURN_NOT_OK(set_extfs("storage_type", fs_config_.storage_type));
    ARROW_RETURN_NOT_OK(set_extfs("cloud_provider", fs_config_.cloud_provider));
    ARROW_RETURN_NOT_OK(set_extfs("address", fs_config_.address));
    ARROW_RETURN_NOT_OK(set_extfs("bucket_name", fs_config_.bucket_name));
    ARROW_RETURN_NOT_OK(set_extfs("region", fs_config_.region));
    ARROW_RETURN_NOT_OK(set_extfs("root_path", fs_config_.root_path));
    ARROW_RETURN_NOT_OK(set_extfs("use_ssl", fs_config_.use_ssl ? "true" : "false"));
    ARROW_RETURN_NOT_OK(set_extfs("use_iam", fs_config_.use_iam ? "true" : "false"));
    if (!fs_config_.access_key_id.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("access_key_id", fs_config_.access_key_id));
    }
    if (!fs_config_.access_key_value.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("access_key_value", fs_config_.access_key_value));
    }
    if (!fs_config_.role_arn.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("role_arn", fs_config_.role_arn));
    }
    if (!fs_config_.session_name.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("session_name", fs_config_.session_name));
    }
    if (!fs_config_.external_id.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("external_id", fs_config_.external_id));
    }
    if (fs_config_.load_frequency > 0) {
      ARROW_RETURN_NOT_OK(set_extfs("load_frequency", std::to_string(fs_config_.load_frequency)));
    }
    if (!fs_config_.gcp_target_service_account.empty()) {
      ARROW_RETURN_NOT_OK(set_extfs("gcp_target_service_account", fs_config_.gcp_target_service_account));
    }
    return arrow::Status::OK();
  }

  arrow::Result<std::shared_ptr<api::ColumnGroups>> WriteIcebergStressData() const {
    ARROW_ASSIGN_OR_RAISE(auto table_uri, MakeIcebergTableUri(base_path_ + "/iceberg"));

    iceberg::IcebergTestTableInfo table_info;
    std::vector<iceberg::IcebergFileInfo> file_infos;
    try {
      auto storage_options = iceberg::ToStorageOptions(fs_config_);
      table_info = iceberg::CreateTestTable(table_uri, kExpectedRows, false, {}, storage_options);
      file_infos = iceberg::PlanFiles(table_info.metadata_location, table_info.snapshot_id, storage_options);
    } catch (const std::exception& e) {
      return arrow::Status::IOError("Failed to create Iceberg stress table: ", e.what());
    }

    if (file_infos.empty()) {
      return arrow::Status::Invalid("Iceberg PlanFiles returned no files");
    }

    auto column_group = std::make_shared<api::ColumnGroup>();
    column_group->columns = {"id", "name", "value"};
    column_group->format = LOON_FORMAT_ICEBERG_TABLE;
    for (const auto& file_info : file_infos) {
      std::unordered_map<std::string, std::string> file_properties;
      if (!file_info.delete_metadata_json.empty()) {
        file_properties[api::kPropertyMetadata] =
            fs_config_.storage_type == "local"
                ? std::string(file_info.delete_metadata_json.begin(), file_info.delete_metadata_json.end())
                : iceberg::ConvertDeleteMetadataPaths(file_info.delete_metadata_json, fs_config_.address);
      }

      auto data_file_path = file_info.data_file_path;
      if (fs_config_.storage_type == "local") {
        data_file_path = LocalReadablePath(data_file_path);
      } else {
        data_file_path = iceberg::ToMilvusUri(data_file_path, fs_config_.address);
      }
      column_group->files.emplace_back(api::ColumnGroupFile{
          std::move(data_file_path), 0, static_cast<int64_t>(file_info.record_count), std::move(file_properties)});
    }

    auto column_groups = std::make_shared<api::ColumnGroups>();
    column_groups->emplace_back(std::move(column_group));
    return column_groups;
  }

  arrow::Result<std::string> MakeIcebergTableUri(const std::string& relative_path) const {
    if (fs_config_.storage_type == "local") {
      return AbsoluteLocalPath(relative_path);
    }
    if (fs_config_.bucket_name.empty()) {
      return arrow::Status::Invalid("BUCKET_NAME env var must be set for remote Iceberg stress test");
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

  arrow::Result<std::vector<int64_t>> RunConcurrentReadStress(api::Reader* reader) const {
    if (!reader) {
      return arrow::Status::Invalid("Cannot run stress read with null reader");
    }

    int threads_ready = 0;
    bool start = false;
    std::chrono::steady_clock::time_point deadline;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<arrow::Status> statuses(kThreadCount, arrow::Status::OK());
    std::vector<int64_t> iterations(kThreadCount, 0);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
      threads.emplace_back([&, thread_index]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          ++threads_ready;
          cv.notify_all();
          cv.wait(lock, [&]() { return start; });
        }

        int64_t iteration = 0;
        do {
          auto status =
              ReadProjectedChunks(reader, StressProjection(thread_index, iteration), kExpectedRows, kReadParallelism);
          if (!status.ok()) {
            statuses[thread_index] = std::move(status);
            break;
          }
          ++iteration;
        } while (std::chrono::steady_clock::now() < deadline);
        iterations[thread_index] = iteration;
      });
    }

    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() { return threads_ready == kThreadCount; });
      deadline = std::chrono::steady_clock::now() + kStressDuration;
      start = true;
    }
    cv.notify_all();

    for (auto& thread : threads) {
      thread.join();
    }

    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
      if (!statuses[thread_index].ok()) {
        return arrow::Status::Invalid("stress thread ", thread_index, " failed after ", iterations[thread_index],
                                      " iterations: ", statuses[thread_index].ToString());
      }
      if (iterations[thread_index] <= 0) {
        return arrow::Status::Invalid("stress thread ", thread_index, " completed no read iterations");
      }
    }

    return iterations;
  }

  static constexpr int kBatchCount = 4;
  static constexpr int64_t kRowsPerBatch = 128;
  static constexpr int64_t kExpectedRows = kBatchCount * kRowsPerBatch;
  static constexpr int kThreadCount = 8;
  static constexpr auto kStressDuration = std::chrono::seconds(30);
  static constexpr size_t kReadParallelism = 4;

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string base_path_;
  api::Properties properties_;
  ArrowFileSystemConfig fs_config_;
};

TEST_P(FormatReaderMetadataCacheParamTest, FormatReaderMetadataCachesOwnsTypedCache) {
  TestMetadataCaches caches;

  ASSERT_STATUS_OK(VisitFormatReaderMetadataCachesForFormat(GetParam(), caches, [&](const auto& typed_cache) {
    using CacheT = std::decay_t<decltype(*typed_cache)>;
    using ReaderT = typename CacheT::ReaderType;
    auto typed_cache_again = caches.get<ReaderT>();

    if (!typed_cache || !typed_cache_again) {
      return arrow::Status::Invalid("format reader metadata caches returned null typed cache");
    }
    EXPECT_EQ(typed_cache.get(), typed_cache_again.get());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, GlobalMetadataCacheReturnsStableTypedCache) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [&](const auto& typed_cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*typed_cache)>;
    using ReaderT = typename CacheT::ReaderType;
    auto typed_cache_again = GetGlobalFormatReaderMetadataCache<ReaderT>();

    if (!typed_cache || !typed_cache_again) {
      return arrow::Status::Invalid("global metadata cache returned null typed cache");
    }
    EXPECT_EQ(typed_cache.get(), typed_cache_again.get());
    return arrow::Status::OK();
  }));
}

TEST(FormatReaderMetadataCacheTest, GlobalCompletedMetadataCacheDoesNotCastWrongReaderType) {
  InitMetadataCache(kDefaultTestMetadataCacheCapacity);
  ClearMetadataCache();

  using ParquetCache = FormatReaderMetadataCache<parquet::ParquetFormatReader>;

  auto parquet_cache = GetGlobalFormatReaderMetadataCache<parquet::ParquetFormatReader>();
  auto vortex_cache = GetGlobalFormatReaderMetadataCache<vortex::VortexFormatReader>();
  const std::string key = "same-raw-key";
  auto parquet_metadata = MakeMetadata<ParquetCache>(key);

  ASSERT_STATUS_OK(parquet_cache->add(key, parquet_metadata));
  auto cached_parquet = parquet_cache->get(key);
  ASSERT_TRUE(cached_parquet.has_value());
  EXPECT_EQ(cached_parquet.value().get(), parquet_metadata.get());
  EXPECT_FALSE(vortex_cache->get(key).has_value());

  ClearMetadataCache();
}

TEST_P(FormatReaderMetadataCacheParamTest, GetOrOpenCachesLoaderResult) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "metadata-cache-hit";
    auto metadata = MakeMetadata<CacheT>(key);
    int loader_calls = 0;

    ARROW_ASSIGN_OR_RAISE(auto first_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return metadata;
                          }));
    ARROW_ASSIGN_OR_RAISE(auto second_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return MakeMetadata<CacheT>("unexpected");
                          }));

    EXPECT_EQ(loader_calls, 1);
    EXPECT_EQ(first_metadata.get(), metadata.get());
    EXPECT_EQ(second_metadata.get(), metadata.get());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, ClearMetadataCacheCausesReload) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "clear-metadata-cache";
    auto first = MakeMetadata<CacheT>(key);
    auto second = MakeMetadata<CacheT>(key);
    int loader_calls = 0;

    ARROW_ASSIGN_OR_RAISE(auto first_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return first;
                          }));

    ClearMetadataCache();

    ARROW_ASSIGN_OR_RAISE(auto second_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return second;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(first_metadata.get(), first.get());
    EXPECT_EQ(second_metadata.get(), second.get());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, InitMetadataCacheZeroPreventsCaching) {
  InitMetadataCache(0);
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "zero-capacity-cache";
    auto first = MakeMetadata<CacheT>(key);
    auto second = MakeMetadata<CacheT>(key);
    int loader_calls = 0;

    ARROW_ASSIGN_OR_RAISE(auto first_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return first;
                          }));
    ARROW_ASSIGN_OR_RAISE(auto second_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return second;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(first_metadata.get(), first.get());
    EXPECT_EQ(second_metadata.get(), second.get());
    EXPECT_FALSE(cache->get(key).has_value());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, ZeroCacheSizeMetadataReloads) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "zero-size-metadata";
    auto first = MakeMetadata<CacheT>(key, 0);
    auto second = MakeMetadata<CacheT>(key, 0);
    int loader_calls = 0;

    ARROW_ASSIGN_OR_RAISE(auto first_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return first;
                          }));
    ARROW_ASSIGN_OR_RAISE(auto second_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return second;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(first_metadata.get(), first.get());
    EXPECT_EQ(second_metadata.get(), second.get());
    EXPECT_FALSE(cache->get(key).has_value());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, AddZeroCacheSizeMetadataClearsExistingEntry) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "add-zero-size-clears-existing";
    auto cached = MakeMetadata<CacheT>(key);
    auto uncached = MakeMetadata<CacheT>(key, 0);

    ARROW_RETURN_NOT_OK(cache->add(key, cached));
    EXPECT_TRUE(cache->get(key).has_value());

    ARROW_RETURN_NOT_OK(cache->add(key, uncached));
    EXPECT_FALSE(cache->get(key).has_value());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, OversizedMetadataReloads) {
  InitMetadataCache(1);
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "oversized-metadata";
    auto first = MakeMetadata<CacheT>(key, 2);
    auto second = MakeMetadata<CacheT>(key, 2);
    int loader_calls = 0;

    ARROW_ASSIGN_OR_RAISE(auto first_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return first;
                          }));
    ARROW_ASSIGN_OR_RAISE(auto second_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return second;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(first_metadata.get(), first.get());
    EXPECT_EQ(second_metadata.get(), second.get());
    EXPECT_FALSE(cache->get(key).has_value());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, AddRejectsNullMetadataAndLeavesKeyAbsent) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "null-add";
    typename CacheT::MetadataPtr null_metadata;

    auto status = cache->add(key, null_metadata);

    EXPECT_TRUE(status.IsInvalid()) << status.ToString();
    EXPECT_FALSE(cache->get(key).has_value());
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, GetOrOpenErrorDoesNotPoisonCache) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "error-retry";
    auto metadata = MakeMetadata<CacheT>(key);
    int loader_calls = 0;

    auto failed = cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
      ++loader_calls;
      return arrow::Status::Invalid("load failed");
    });
    EXPECT_TRUE(failed.status().IsInvalid()) << failed.status().ToString();
    EXPECT_FALSE(cache->get(key).has_value());

    ARROW_ASSIGN_OR_RAISE(auto retried_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return metadata;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(retried_metadata.get(), metadata.get());
    auto cached = cache->get(key);
    EXPECT_TRUE(cached.has_value());
    if (cached.has_value()) {
      EXPECT_EQ(cached.value().get(), metadata.get());
    }
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, GetOrOpenNullMetadataDoesNotPoisonCache) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    const std::string key = "null-retry";
    auto metadata = MakeMetadata<CacheT>(key);
    int loader_calls = 0;

    auto failed = cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
      ++loader_calls;
      return typename CacheT::MetadataPtr{};
    });
    EXPECT_TRUE(failed.status().IsInvalid()) << failed.status().ToString();
    EXPECT_FALSE(cache->get(key).has_value());

    ARROW_ASSIGN_OR_RAISE(auto retried_metadata,
                          cache->get_or_open(key, [&]() -> arrow::Result<typename CacheT::MetadataPtr> {
                            ++loader_calls;
                            return metadata;
                          }));

    EXPECT_EQ(loader_calls, 2);
    EXPECT_EQ(retried_metadata.get(), metadata.get());
    auto cached = cache->get(key);
    EXPECT_TRUE(cached.has_value());
    if (cached.has_value()) {
      EXPECT_EQ(cached.value().get(), metadata.get());
    }
    return arrow::Status::OK();
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, GetOrOpenThrowingLoaderNotifiesWaitersAndAllowsRetry) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    return RunThrowingLoaderNotifiesWaitersAndAllowsRetry<CacheT>(cache);
  }));
}

TEST_P(FormatReaderMetadataCacheParamTest, GetOrOpenSingleflightsConcurrentSameKey) {
  ASSERT_STATUS_OK(VisitGlobalMetadataCacheForFormat(GetParam(), [](const auto& cache) -> arrow::Status {
    using CacheT = std::decay_t<decltype(*cache)>;
    return RunSingleflightsConcurrentSameKey<CacheT>(cache);
  }));
}

TEST(FormatReaderMetadataCacheTest, FormatCacheKeysKeepFormatSpecificIdentity) {
  api::ColumnGroupFile file;
  file.path = "s3://bucket/path/data-file";
  file.properties[api::kPropertyFileSize] = "1024";
  file.properties[api::kPropertyFooterSize] = "128";
  file.properties[api::kPropertyMetadata] = "manifest-metadata";

  EXPECT_NE(parquet::ParquetFormatReader::MetaTrait::cache_key(file), file.path);
  EXPECT_NE(vortex::VortexFormatReader::MetaTrait::cache_key(file), file.path);
  EXPECT_NE(lance::LanceTableReader::MetaTrait::cache_key(file), file.path);
  EXPECT_NE(iceberg::IcebergFormatReader::MetaTrait::cache_key(file), file.path);
}

TEST(FormatReaderMetadataCacheTest, LanceCacheKeyIncludesTableVersionWhenPresent) {
  api::ColumnGroupFile file;
  file.path = "s3://bucket/path/table?fragment_id=7";
  file.properties[api::kPropertyFileSize] = "0";
  file.properties[api::kPropertyFooterSize] = "0";

  file.properties[lance::kLanceTableVersionProperty] = "3";
  auto version_3_key = lance::LanceTableReader::MetaTrait::cache_key(file);

  file.properties[lance::kLanceTableVersionProperty] = "4";
  auto version_4_key = lance::LanceTableReader::MetaTrait::cache_key(file);

  EXPECT_NE(version_3_key, version_4_key);
  EXPECT_NE(version_3_key.find("lance.table.version:3"), std::string::npos);
  EXPECT_NE(version_4_key.find("lance.table.version:4"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Formats,
    FormatReaderMetadataCacheParamTest,
    ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE));

TEST_P(FormatReaderMetadataCacheStressTest, ConcurrentReaderCacheOpenAndRead) {
  const auto format = GetParam();
  const auto* use_azurite = std::getenv("USE_AZURITE");
  if (format == LOON_FORMAT_ICEBERG_TABLE && fs_config_.cloud_provider == kCloudProviderAzure && use_azurite &&
      std::string(use_azurite) == "true") {
    GTEST_SKIP() << "Iceberg test table creation does not support Azurite endpoint normalization";
  }

  ASSERT_AND_ASSIGN(auto cgs, WriteStressData(format));
  ASSERT_EQ(cgs->size(), 1);
  ASSERT_NE((*cgs)[0], nullptr);
  ASSERT_FALSE((*cgs)[0]->files.empty());

  auto reader = api::Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);

  ASSERT_AND_ASSIGN(auto iterations, RunConcurrentReadStress(reader.get()));
  EXPECT_EQ(iterations.size(), static_cast<size_t>(kThreadCount));
}

TEST_P(FormatReaderMetadataCacheStressTest, ReaderDisabledMetadataCacheBypassesGlobalCache) {
  const auto format = GetParam();
  const auto* use_azurite = std::getenv("USE_AZURITE");
  if (format == LOON_FORMAT_ICEBERG_TABLE && fs_config_.cloud_provider == kCloudProviderAzure && use_azurite &&
      std::string(use_azurite) == "true") {
    GTEST_SKIP() << "Iceberg test table creation does not support Azurite endpoint normalization";
  }

  ASSERT_AND_ASSIGN(auto cgs, WriteStressData(format));
  ASSERT_EQ(cgs->size(), 1);
  ASSERT_NE((*cgs)[0], nullptr);
  ASSERT_FALSE((*cgs)[0]->files.empty());

  api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "true");
  auto enabled_reader = api::Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(enabled_reader, nullptr);
  ASSERT_STATUS_OK(ReadProjectedChunks(enabled_reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));

  auto disabled_properties = properties_;
  api::SetValue(disabled_properties, PROPERTY_READER_METADATA_CACHE_ENABLE, "false");
  api::SetValue(disabled_properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "128");
  auto disabled_reader = api::Reader::create(cgs, schema_, nullptr, disabled_properties);
  ASSERT_NE(disabled_reader, nullptr);
  auto projected_columns = std::make_shared<std::vector<std::string>>();
  projected_columns->emplace_back("id");
  ASSERT_AND_ASSIGN(auto disabled_chunk_reader, disabled_reader->get_chunk_reader(0, projected_columns));
  if (format != LOON_FORMAT_ICEBERG_TABLE) {
    EXPECT_EQ(disabled_chunk_reader->total_number_of_chunks(), static_cast<size_t>((kExpectedRows + 127) / 128));
  }
  ASSERT_STATUS_OK(ReadProjectedChunks(disabled_reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));
}

TEST_P(FormatReaderMetadataCacheStressTest, ReaderMetadataCacheUsesCurrentProperties) {
  const auto format = GetParam();
  if (format != LOON_FORMAT_VORTEX && format != LOON_FORMAT_LANCE_TABLE) {
    GTEST_SKIP() << "logical chunk rows only changes cached reader chunking for Vortex and Lance";
  }

  ASSERT_AND_ASSIGN(auto cgs, WriteStressData(format));
  ASSERT_EQ(cgs->size(), 1);
  ASSERT_NE((*cgs)[0], nullptr);
  ASSERT_FALSE((*cgs)[0]->files.empty());

  api::SetValue(properties_, PROPERTY_READER_METADATA_CACHE_ENABLE, "true");
  api::SetValue(properties_, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "512");
  auto warm_reader = api::Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(warm_reader, nullptr);
  ASSERT_STATUS_OK(ReadProjectedChunks(warm_reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));

  auto changed_properties = properties_;
  api::SetValue(changed_properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS, "128");
  auto changed_reader = api::Reader::create(cgs, schema_, nullptr, changed_properties);
  ASSERT_NE(changed_reader, nullptr);
  auto projected_columns = std::make_shared<std::vector<std::string>>();
  projected_columns->emplace_back("id");
  ASSERT_AND_ASSIGN(auto changed_chunk_reader, changed_reader->get_chunk_reader(0, projected_columns));
  EXPECT_EQ(changed_chunk_reader->total_number_of_chunks(), static_cast<size_t>((kExpectedRows + 127) / 128));
  ASSERT_STATUS_OK(ReadProjectedChunks(changed_reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));
}

TEST_P(FormatReaderMetadataCacheStressTest, CachedMetadataUsesCurrentFilesystemProperties) {
  const auto format = GetParam();
  if (format != LOON_FORMAT_PARQUET && format != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Parquet reopens with current properties; Vortex separates cached handles by filesystem identity";
  }
  if (!IsLocalFileSystem(fs_)) {
    GTEST_SKIP() << "local root_path switch is required to prove current filesystem properties are used";
  }

  ASSERT_AND_ASSIGN(auto cgs, WriteStressData(format));
  ASSERT_EQ(cgs->size(), 1);
  ASSERT_NE((*cgs)[0], nullptr);
  ASSERT_FALSE((*cgs)[0]->files.empty());

  auto warm_reader = api::Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(warm_reader, nullptr);
  ASSERT_STATUS_OK(ReadProjectedChunks(warm_reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));

  auto changed_properties = properties_;
  const auto missing_root = std::filesystem::path("/tmp") / ("milvus-storage-cache-missing-root-" + format);
  std::filesystem::remove_all(missing_root);
  api::SetValue(changed_properties, PROPERTY_FS_ROOT_PATH, missing_root.string().c_str());

  auto changed_reader = api::Reader::create(cgs, schema_, nullptr, changed_properties);
  ASSERT_NE(changed_reader, nullptr);
  auto status = ReadProjectedChunks(changed_reader.get(), {"id"}, kExpectedRows, kReadParallelism);
  EXPECT_FALSE(status.ok()) << "metadata cache reused a filesystem object from the warm reader";

  std::filesystem::remove_all(missing_root);
}

TEST_P(FormatReaderMetadataCacheStressTest, VortexCachesMetadataWithoutManifestFileSizes) {
  const auto format = GetParam();
  if (format != LOON_FORMAT_VORTEX) {
    GTEST_SKIP() << "Vortex should cache metadata after resolving missing file sizes";
  }

  ASSERT_AND_ASSIGN(auto cgs, WriteStressData(format));
  ASSERT_EQ(cgs->size(), 1);
  ASSERT_NE((*cgs)[0], nullptr);
  ASSERT_FALSE((*cgs)[0]->files.empty());

  std::vector<uint64_t> expected_footer_sizes;
  expected_footer_sizes.reserve((*cgs)[0]->files.size());
  for (auto& file : (*cgs)[0]->files) {
    expected_footer_sizes.emplace_back(file.Get<uint64_t>(api::kPropertyFooterSize));
    file.properties.erase(api::kPropertyFileSize);
    file.properties.erase(api::kPropertyFooterSize);
  }

  auto reader = api::Reader::create(cgs, schema_, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_STATUS_OK(ReadProjectedChunks(reader.get(), {"id"}, kExpectedRows, kReadParallelism));
  ASSERT_STATUS_OK(ExpectGlobalMetadataCacheEntriesForFormat(cgs, format, properties_, true));

  auto cache = GetGlobalFormatReaderMetadataCache<vortex::VortexFormatReader>();
  for (size_t i = 0; i < (*cgs)[0]->files.size(); ++i) {
    const auto& file = (*cgs)[0]->files[i];
    ASSERT_GT(expected_footer_sizes[i], 0u);
    ASSERT_AND_ASSIGN(auto key, MakeFormatReaderMetadataCacheKey(
                                    file, properties_, vortex::VortexFormatReader::MetaTrait::cache_key(file)));
    auto cached = cache->get(key);
    ASSERT_TRUE(cached.has_value()) << "Expected cached metadata for key: " << key;
    EXPECT_EQ(cached.value()->cache_size, expected_footer_sizes[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Formats,
    FormatReaderMetadataCacheStressTest,
    ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE, LOON_FORMAT_ICEBERG_TABLE));

}  // namespace milvus_storage::test
