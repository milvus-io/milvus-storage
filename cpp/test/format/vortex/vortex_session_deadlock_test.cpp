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

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/filesystem/filesystem.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/thread_pool.h"
#include "test_env.h"

namespace milvus_storage::vortex {

namespace {

constexpr char kReproRootEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_ROOT";
constexpr char kReproFormatVersionEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_FORMAT_VERSION";
constexpr char kReproStatsEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_STATS";
constexpr char kReproReadGroupsEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_READ_GROUPS";
constexpr char kReproRuntimeCpuEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_RUNTIME_CPU";
constexpr char kReproRuntimeIoEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_RUNTIME_IO";
constexpr char kReproReadParallelismEnv[] = "MILVUS_STORAGE_VORTEX_DEADLOCK_READ_PARALLELISM";
constexpr char kChildStorageTypeEnv[] = "TEST_ENV_STORAGE_TYPE";

constexpr int64_t kRows = 5000;
constexpr int64_t kBatchRows = 100;
constexpr int64_t kDimension = 36;
constexpr uint32_t kIssueSeed = 20260806;
constexpr auto kWriteTimeout = std::chrono::seconds(30);
constexpr auto kReadTimeout = std::chrono::seconds(5);

constexpr char kGroup0Path[] = "group0.vortex";
constexpr char kGroup1Path[] = "group1.vortex";
constexpr char kGroup2Path[] = "group2.vortex";

std::string GetEnv(const char* name) {
  const auto* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

uint32_t ReproFormatVersion() {
  const auto value = GetEnv(kReproFormatVersionEnv);
  return value.empty() ? 2 : static_cast<uint32_t>(std::stoul(value));
}

bool ReproStatsEnabled() {
  const auto value = GetEnv(kReproStatsEnv);
  return value.empty() || value == "1" || value == "true";
}

uint32_t ReproRuntimeThreads(const char* env_name, uint32_t default_value) {
  const auto value = GetEnv(env_name);
  return value.empty() ? default_value : static_cast<uint32_t>(std::stoul(value));
}

size_t ReproReadParallelism() {
  const auto value = GetEnv(kReproReadParallelismEnv);
  return value.empty() ? 1 : static_cast<size_t>(std::stoul(value));
}

std::vector<size_t> ReproReadGroups() {
  const auto value = GetEnv(kReproReadGroupsEnv);
  const auto groups = value.empty() ? std::string("012") : value;
  std::vector<size_t> result;
  for (const auto character : groups) {
    if (character < '0' || character > '2') {
      continue;
    }
    const auto group = static_cast<size_t>(character - '0');
    if (std::find(result.begin(), result.end(), group) == result.end()) {
      result.emplace_back(group);
    }
  }
  return result;
}

std::shared_ptr<arrow::KeyValueMetadata> FieldMetadata(int64_t field_id,
                                                       std::optional<int64_t> dimension = std::nullopt) {
  std::vector<std::string> keys{"PARQUET:field_id"};
  std::vector<std::string> values{std::to_string(field_id)};
  if (dimension.has_value()) {
    keys.emplace_back("dim");
    values.emplace_back(std::to_string(*dimension));
  }
  return arrow::key_value_metadata(std::move(keys), std::move(values));
}

arrow::Result<std::shared_ptr<arrow::Schema>> MakeIssueSchema() {
  // This is the schema passed by Milvus core to milvus-storage: physical
  // column names are field IDs.  Nullable dense vectors are represented as
  // Arrow binary values (one 36*4-byte value per non-null row), with dim in
  // field metadata.
  return arrow::schema({
      arrow::field("100", arrow::int64(), false, FieldMetadata(100)),
      arrow::field("0", arrow::int64(), false, FieldMetadata(0)),
      arrow::field("1", arrow::int64(), false, FieldMetadata(1)),
      arrow::field("101", arrow::binary(), true, FieldMetadata(101, kDimension)),
      arrow::field("102", arrow::int8(), true, FieldMetadata(102)),
  });
}

std::shared_ptr<arrow::Schema> SelectSchema(const std::shared_ptr<arrow::Schema>& schema,
                                            std::initializer_list<const char*> columns) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (const auto* column : columns) {
    fields.emplace_back(schema->GetFieldByName(column));
  }
  return arrow::schema(std::move(fields));
}

struct IssueBatchSet {
  std::shared_ptr<arrow::RecordBatch> group0;
  std::shared_ptr<arrow::RecordBatch> group1;
  std::shared_ptr<arrow::RecordBatch> group2;
};

std::string NormalizedVectorBytes(std::mt19937& rng) {
  std::array<float, kDimension> values{};
  std::normal_distribution<float> normal(0.0F, 1.0F);
  float norm_squared = 0.0F;
  for (auto& value : values) {
    value = normal(rng);
    norm_squared += value * value;
  }
  const auto norm = std::sqrt(norm_squared);
  for (auto& value : values) {
    value /= norm;
  }
  return std::string(reinterpret_cast<const char*>(values.data()), sizeof(values));
}

arrow::Result<IssueBatchSet> MakeIssueBatchSet(const std::shared_ptr<arrow::Schema>& group0_schema,
                                               const std::shared_ptr<arrow::Schema>& group1_schema,
                                               const std::shared_ptr<arrow::Schema>& group2_schema,
                                               int64_t batch_index,
                                               std::mt19937& issue_rng,
                                               std::mt19937& vector_rng) {
  arrow::Int64Builder ids;
  arrow::Int64Builder row_ids;
  arrow::Int64Builder timestamps;
  arrow::Int8Builder int8_values;
  arrow::BinaryBuilder vectors;

  std::uniform_int_distribution<int> int8_distribution(-128, 127);
  std::uniform_real_distribution<float> probability(0.0F, 1.0F);
  const auto int8_value = static_cast<int8_t>(int8_distribution(issue_rng));

  // NumPy creates the complete matrix before the Python list comprehension.
  // Generate every vector, including rows that will become null, to preserve
  // the same data shape and RNG progression.
  std::array<std::string, kBatchRows> vector_values;
  for (auto& value : vector_values) {
    value = NormalizedVectorBytes(vector_rng);
  }

  const auto start_row = batch_index * kBatchRows;
  for (int64_t row = 0; row < kBatchRows; ++row) {
    const auto global_row = start_row + row;
    ARROW_RETURN_NOT_OK(ids.Append(global_row));
    ARROW_RETURN_NOT_OK(row_ids.Append(global_row));
    ARROW_RETURN_NOT_OK(timestamps.Append(1'000'000 + global_row));

    if (row % 10 == 9) {
      ARROW_RETURN_NOT_OK(vectors.AppendNull());
    } else {
      ARROW_RETURN_NOT_OK(vectors.Append(vector_values[static_cast<size_t>(row)]));
    }

    // The issue uses random.random() < 0.8 independently for every INT8
    // value, including rows whose vector is null.
    if (probability(issue_rng) < 0.8F) {
      ARROW_RETURN_NOT_OK(int8_values.Append(int8_value));
    } else {
      ARROW_RETURN_NOT_OK(int8_values.AppendNull());
    }
  }

  std::shared_ptr<arrow::Array> id_array;
  std::shared_ptr<arrow::Array> row_id_array;
  std::shared_ptr<arrow::Array> timestamp_array;
  std::shared_ptr<arrow::Array> vector_array;
  std::shared_ptr<arrow::Array> int8_array;
  ARROW_RETURN_NOT_OK(ids.Finish(&id_array));
  ARROW_RETURN_NOT_OK(row_ids.Finish(&row_id_array));
  ARROW_RETURN_NOT_OK(timestamps.Finish(&timestamp_array));
  ARROW_RETURN_NOT_OK(vectors.Finish(&vector_array));
  ARROW_RETURN_NOT_OK(int8_values.Finish(&int8_array));

  return IssueBatchSet{
      .group0 = arrow::RecordBatch::Make(group0_schema, kBatchRows, {id_array, row_id_array, timestamp_array}),
      .group1 = arrow::RecordBatch::Make(group1_schema, kBatchRows, {int8_array}),
      .group2 = arrow::RecordBatch::Make(group2_schema, kBatchRows, {vector_array}),
  };
}

arrow::Status ConfigureReproProperties(api::Properties& properties) {
  const auto root = GetEnv(kReproRootEnv);
  if (root.empty()) {
    return arrow::Status::Invalid("missing ", kReproRootEnv);
  }

  ARROW_RETURN_NOT_OK(InitTestProperties(properties));
  api::SetValue(properties, PROPERTY_FS_ROOT_PATH, root.c_str());
  api::SetValue(properties, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS, ReproStatsEnabled() ? "true" : "false");
  api::SetValue(properties, PROPERTY_WRITER_VORTEX_FORMAT_VERSION, std::to_string(ReproFormatVersion()).c_str());
  api::SetValue(properties, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE, std::to_string(128 * 1024).c_str());
  // Keep the reader's logical chunking equal to the issue's single sealed
  // segment.  The physical vector file can still contain several Vortex row
  // groups; ColumnGroupReader will merge them through its normal path.
  api::SetValue(properties, PROPERTY_READER_LOGICAL_CHUNK_ROWS, std::to_string(kRows).c_str());
  return arrow::Status::OK();
}

void PinToOneAllowedCpu() {
#if !defined(__linux__)
  return;
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    return;
  }

  int selected_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      selected_cpu = cpu;
      break;
    }
  }
  if (selected_cpu < 0) {
    return;
  }

  cpu_set_t one_cpu;
  CPU_ZERO(&one_cpu);
  CPU_SET(selected_cpu, &one_cpu);
  (void)sched_setaffinity(0, sizeof(one_cpu), &one_cpu);
#endif
}

api::ColumnGroup MakeColumnGroup(std::initializer_list<const char*> columns,
                                 const char* path,
                                 int64_t file_size,
                                 int64_t footer_size) {
  api::ColumnGroup group;
  for (const auto* column : columns) {
    group.columns.emplace_back(column);
  }
  group.format = LOON_FORMAT_VORTEX;
  api::ColumnGroupFile file;
  file.path = path;
  file.start_index = 0;
  file.end_index = kRows;
  file.Set(api::kPropertyFileSize, file_size);
  file.Set(api::kPropertyFooterSize, footer_size);
  group.files.emplace_back(std::move(file));
  return group;
}

TEST(VortexSessionDeadlockTest, DISABLED_ChildWrite) {
  setenv(kChildStorageTypeEnv, "local", 1);

  api::Properties properties;
  ASSERT_STATUS_OK(ConfigureReproProperties(properties));
  ASSERT_AND_ASSIGN(auto file_system, GetFileSystem(properties));
  ASSERT_AND_ASSIGN(auto schema, MakeIssueSchema());
  const auto group0_schema = SelectSchema(schema, {"100", "0", "1"});
  const auto group1_schema = SelectSchema(schema, {"102"});
  const auto group2_schema = SelectSchema(schema, {"101"});

  std::fprintf(stderr, "vortex issue reproducer: writing 3 groups (%lld rows, V%d)\n", static_cast<long long>(kRows),
               ReproFormatVersion());
  std::fflush(stderr);

  ASSERT_AND_ASSIGN(auto group0_writer, VortexFileWriter::Open(file_system, group0_schema, kGroup0Path, properties));
  ASSERT_AND_ASSIGN(auto group1_writer, VortexFileWriter::Open(file_system, group1_schema, kGroup1Path, properties));
  ASSERT_AND_ASSIGN(auto group2_writer, VortexFileWriter::Open(file_system, group2_schema, kGroup2Path, properties));

  std::mt19937 issue_rng(kIssueSeed);
  std::mt19937 vector_rng(kIssueSeed);
  for (int64_t batch = 0; batch < kRows / kBatchRows; ++batch) {
    ASSERT_AND_ASSIGN(auto batches,
                      MakeIssueBatchSet(group0_schema, group1_schema, group2_schema, batch, issue_rng, vector_rng));
    ASSERT_STATUS_OK(group0_writer->Write(batches.group0));
    ASSERT_STATUS_OK(group1_writer->Write(batches.group1));
    ASSERT_STATUS_OK(group2_writer->Write(batches.group2));
  }

  ASSERT_STATUS_OK(group0_writer->Flush());
  ASSERT_STATUS_OK(group1_writer->Flush());
  ASSERT_STATUS_OK(group2_writer->Flush());
  ASSERT_AND_ASSIGN(auto group0_file, group0_writer->Close());
  ASSERT_AND_ASSIGN(auto group1_file, group1_writer->Close());
  ASSERT_AND_ASSIGN(auto group2_file, group2_writer->Close());
  ASSERT_EQ(group0_file.end_index, kRows);
  ASSERT_EQ(group1_file.end_index, kRows);
  ASSERT_EQ(group2_file.end_index, kRows);

  std::fprintf(stderr, "vortex issue reproducer: write complete; sizes group0=%lld group1=%lld group2=%lld\n",
               static_cast<long long>(group0_file.Get<int64_t>(api::kPropertyFileSize)),
               static_cast<long long>(group1_file.Get<int64_t>(api::kPropertyFileSize)),
               static_cast<long long>(group2_file.Get<int64_t>(api::kPropertyFileSize)));
  std::fflush(stderr);
}

struct ChildWaitResult {
  bool exited = false;
  int status = 0;
};

ChildWaitResult WaitForChild(pid_t pid, std::chrono::milliseconds timeout) {
  ChildWaitResult result;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const auto waited = waitpid(pid, &result.status, WNOHANG);
    if (waited == pid) {
      result.exited = true;
      return result;
    }
    if (waited < 0 && errno != EINTR) {
      return result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      (void)kill(pid, SIGKILL);
      while (waitpid(pid, &result.status, 0) < 0 && errno == EINTR) {
      }
      return result;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

pid_t SpawnChild(const char* filter, const std::string& root, uint32_t format_version) {
#if defined(__APPLE__)
  uint32_t executable_path_size = 0;
  (void)_NSGetExecutablePath(nullptr, &executable_path_size);
  std::string executable_path(executable_path_size, '\0');
  if (_NSGetExecutablePath(executable_path.data(), &executable_path_size) != 0) {
    errno = ENAMETOOLONG;
    return -1;
  }
#else
  const std::string executable_path = "/proc/self/exe";
#endif

  const auto pid = fork();
  if (pid != 0) {
    return pid;
  }

  setenv(kChildStorageTypeEnv, "local", 1);
  setenv(kReproRootEnv, root.c_str(), 1);
  setenv(kReproFormatVersionEnv, std::to_string(format_version).c_str(), 1);

  execl(executable_path.c_str(), "milvus_test", filter, "--gtest_color=no", "--gtest_also_run_disabled_tests",
        static_cast<char*>(nullptr));
  _exit(127);
}

std::string MakeTemporaryRoot() {
  const auto base =
      std::filesystem::temp_directory_path() / ("milvus-storage-vortex-session-deadlock-" + std::to_string(getpid()));
  std::error_code error;
  std::filesystem::remove_all(base, error);
  std::filesystem::create_directories(base, error);
  return base.string();
}

TEST(VortexSessionDeadlockTest, DISABLED_ChildRead) {
  setenv(kChildStorageTypeEnv, "local", 1);
  PinToOneAllowedCpu();

  // This must happen before the first Rust/Vortex bridge call.  Restricting
  // both pools makes the session/runtime dependency cycle reproducible and
  // mirrors the constrained local query-node load used by the issue.
  ASSERT_STATUS_OK(
      ConfigureStorageRuntime(ReproRuntimeThreads(kReproRuntimeCpuEnv, 1), ReproRuntimeThreads(kReproRuntimeIoEnv, 1)));

  api::Properties properties;
  ASSERT_STATUS_OK(ConfigureReproProperties(properties));
  ASSERT_AND_ASSIGN(auto file_system, GetFileSystem(properties));
  ASSERT_AND_ASSIGN(auto schema, MakeIssueSchema());

  auto group0 = std::make_shared<api::ColumnGroup>(MakeColumnGroup({"100", "0", "1"}, kGroup0Path, 0, 0));
  auto group1 = std::make_shared<api::ColumnGroup>(MakeColumnGroup({"102"}, kGroup1Path, 0, 0));
  auto group2 = std::make_shared<api::ColumnGroup>(MakeColumnGroup({"101"}, kGroup2Path, 0, 0));
  (void)file_system;  // FilesystemCache resolves the local root from properties.

  // Open all three readers through the same factory used by the storage
  // column-group path.  The file sizes are optional for local files; open()
  // obtains them from the footer when the supplied value is zero.
  ASSERT_AND_ASSIGN(auto reader0,
                    api::ColumnGroupReader::create(schema, group0, {"100", "0", "1"}, properties, nullptr));
  ASSERT_AND_ASSIGN(auto reader1, api::ColumnGroupReader::create(schema, group1, {"102"}, properties, nullptr));
  ASSERT_AND_ASSIGN(auto reader2, api::ColumnGroupReader::create(schema, group2, {"101"}, properties, nullptr));

  std::array<std::unique_ptr<api::ColumnGroupReader>*, 3> readers{&reader0, &reader1, &reader2};
  std::array<std::vector<int64_t>, 3> chunk_indices;
  for (size_t i = 0; i < readers.size(); ++i) {
    const auto count = (*readers[i])->total_number_of_chunks();
    chunk_indices[i].resize(count);
    std::iota(chunk_indices[i].begin(), chunk_indices[i].end(), 0);
    std::fprintf(stderr, "vortex issue reproducer: group%zu opened rows=%zu chunks=%zu\n", i,
                 (*readers[i])->total_rows(), count);
  }
  std::fflush(stderr);

  const auto active_groups = ReproReadGroups();
  ASSERT_FALSE(active_groups.empty()) << "set " << kReproReadGroupsEnv << " to one or more of 0,1,2";
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(active_groups.size() + 1));
  std::array<std::string, 3> errors;
  std::array<size_t, 3> result_counts{};
  std::vector<std::thread> workers;
  workers.reserve(active_groups.size());
  for (const auto i : active_groups) {
    workers.emplace_back([&, i]() {
      start_barrier.arrive_and_wait();
      std::fprintf(stderr, "vortex issue reproducer: group%zu get_chunks start\n", i);
      std::fflush(stderr);
      auto result = (*readers[i])->get_chunks(chunk_indices[i], ReproReadParallelism());
      if (!result.ok()) {
        errors[i] = result.status().ToString();
        return;
      }
      result_counts[i] = result.ValueOrDie().size();
      std::fprintf(stderr, "vortex issue reproducer: group%zu get_chunks complete (%zu batches)\n", i,
                   result_counts[i]);
      std::fflush(stderr);
    });
  }
  start_barrier.arrive_and_wait();
  for (auto& worker : workers) {
    worker.join();
  }

  for (const auto i : active_groups) {
    ASSERT_TRUE(errors[i].empty()) << "group" << i << ": " << errors[i];
    ASSERT_EQ(result_counts[i], chunk_indices[i].size());
  }
}

}  // namespace

TEST(VortexSessionDeadlockTest, IssueColumnGroupsReaderCompletes) {
  const auto root = MakeTemporaryRoot();
  const uint32_t format_version = 2;

  const auto writer_pid =
      SpawnChild("--gtest_filter=VortexSessionDeadlockTest.DISABLED_ChildWrite", root, format_version);
  ASSERT_GT(writer_pid, 0) << "fork failed: " << std::strerror(errno);
  const auto writer_result = WaitForChild(writer_pid, kWriteTimeout);
  ASSERT_TRUE(writer_result.exited) << "writer child timed out";
  ASSERT_TRUE(WIFEXITED(writer_result.status));
  ASSERT_EQ(WEXITSTATUS(writer_result.status), 0);

  const auto reader_pid =
      SpawnChild("--gtest_filter=VortexSessionDeadlockTest.DISABLED_ChildRead", root, format_version);
  ASSERT_GT(reader_pid, 0) << "fork failed: " << std::strerror(errno);
  const auto reader_result = WaitForChild(reader_pid, kReadTimeout);

  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);

  ASSERT_TRUE(reader_result.exited) << "reader child timed out; status=" << reader_result.status;
  ASSERT_TRUE(WIFEXITED(reader_result.status));
  ASSERT_EQ(WEXITSTATUS(reader_result.status), 0);
}

}  // namespace milvus_storage::vortex
