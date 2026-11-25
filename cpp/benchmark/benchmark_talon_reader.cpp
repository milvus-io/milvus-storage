// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/future.h>

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "test_env.h"

namespace milvus_storage {

class TalonReaderBenchmark : public ::benchmark::Fixture {
  protected:
  void SetUp(::benchmark::State& state) override {
    if (!IsTalonEnv()) {
      state.SkipWithError("Talon benchmark environment is not configured");
      return;
    }

    const auto properties_status = InitTestProperties(properties_);
    if (!properties_status.ok()) {
      state.SkipWithError(("Failed to initialize Talon benchmark properties: " + properties_status.ToString()).c_str());
      return;
    }

    FilesystemCache::getInstance().clean();
    auto fs_result = GetFileSystem(properties_);
    if (!fs_result.ok()) {
      state.SkipWithError(("Failed to create Talon benchmark filesystem: " + fs_result.status().ToString()).c_str());
      return;
    }
    fs_ = std::move(fs_result).ValueOrDie();

    const auto unique_suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = "talon-benchmark/reader-" + std::to_string(getpid()) + "-" + std::to_string(unique_suffix);
    expected_.resize(kObjectSize);
    for (size_t i = 0; i < expected_.size(); ++i) {
      expected_[i] = static_cast<uint8_t>(i % 251);
    }

    auto output_result = fs_->OpenOutputStream(path_);
    if (!output_result.ok()) {
      state.SkipWithError(("Failed to open Talon benchmark object: " + output_result.status().ToString()).c_str());
      return;
    }
    auto output = std::move(output_result).ValueOrDie();
    const auto write_status = output->Write(expected_.data(), static_cast<int64_t>(expected_.size()));
    if (!write_status.ok()) {
      (void)output->Close();
      state.SkipWithError(("Failed to write Talon benchmark object: " + write_status.ToString()).c_str());
      return;
    }
    const auto close_status = output->Close();
    if (!close_status.ok()) {
      state.SkipWithError(("Failed to close Talon benchmark object: " + close_status.ToString()).c_str());
      return;
    }

    auto info_result = fs_->GetFileInfo(path_);
    if (!info_result.ok()) {
      state.SkipWithError(("Failed to stat Talon benchmark object: " + info_result.status().ToString()).c_str());
      return;
    }
    const auto info = std::move(info_result).ValueOrDie();
    if (info.type() != arrow::fs::FileType::File || info.size() != static_cast<int64_t>(expected_.size())) {
      state.SkipWithError("Talon benchmark object has an unexpected type or size");
      return;
    }

    auto file_result = fs_->OpenInputFile(info);
    if (!file_result.ok()) {
      state.SkipWithError(("Failed to open Talon benchmark reader: " + file_result.status().ToString()).c_str());
      return;
    }
    file_ = std::move(file_result).ValueOrDie();

    auto warm_result = file_->ReadAt(0, static_cast<int64_t>(expected_.size()));
    if (!warm_result.ok()) {
      state.SkipWithError(("Failed to warm Talon benchmark object: " + warm_result.status().ToString()).c_str());
      return;
    }
    const auto warm = std::move(warm_result).ValueOrDie();
    if (warm->size() != static_cast<int64_t>(expected_.size()) ||
        std::memcmp(warm->data(), expected_.data(), expected_.size()) != 0) {
      state.SkipWithError("Talon benchmark warm read returned unexpected data");
      return;
    }

    scratch_.resize(expected_.size());
    small_io_offsets_.resize(kObjectSize / kSmallIoSize);
    for (size_t i = 0; i < small_io_offsets_.size(); ++i) {
      // 4051 is odd, so multiplication permutes every 4 KiB page in the
      // power-of-two-sized object instead of benchmarking one hot region.
      const size_t page = (i * 4051U) % small_io_offsets_.size();
      small_io_offsets_[i] = static_cast<int64_t>(page * kSmallIoSize);
    }
  }

  void TearDown(::benchmark::State& state) override {
    if (file_ != nullptr) {
      const auto close_status = file_->Close();
      if (!close_status.ok()) {
        state.SkipWithError(("Failed to close Talon benchmark reader: " + close_status.ToString()).c_str());
      }
      file_.reset();
    }
    if (fs_ != nullptr && !path_.empty()) {
      const auto delete_status = fs_->DeleteFile(path_);
      if (!delete_status.ok()) {
        state.SkipWithError(("Failed to delete Talon benchmark object: " + delete_status.ToString()).c_str());
      }
      fs_.reset();
    }
    FilesystemCache::getInstance().clean();
  }

  static constexpr size_t kObjectSize = 64U * 1024U * 1024U;
  static constexpr size_t kRangeSize = 4U * 1024U * 1024U;
  static constexpr size_t kSmallIoSize = 4U * 1024U;
  static constexpr size_t kIopsBatchSize = 64U;

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::string path_;
  std::vector<uint8_t> expected_;
  std::vector<uint8_t> scratch_;
  std::vector<int64_t> small_io_offsets_;
};

BENCHMARK_DEFINE_F(TalonReaderBenchmark, FullObjectReadAt)(::benchmark::State& state) {
  if (file_ == nullptr) {
    return;
  }

  for (auto _ : state) {
    auto read_result = file_->ReadAt(0, static_cast<int64_t>(expected_.size()));
    if (!read_result.ok()) {
      state.SkipWithError(("Talon full-object read failed: " + read_result.status().ToString()).c_str());
      break;
    }
    const auto buffer = std::move(read_result).ValueOrDie();
    if (buffer->size() != static_cast<int64_t>(expected_.size())) {
      state.SkipWithError("Talon full-object read returned an unexpected length");
      break;
    }
    ::benchmark::DoNotOptimize(buffer->data());
  }

  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(expected_.size()));
}

BENCHMARK_DEFINE_F(TalonReaderBenchmark, ConcurrentAsyncRanges)(::benchmark::State& state) {
  if (file_ == nullptr) {
    return;
  }
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file_.get());
  if (async_file == nullptr) {
    state.SkipWithError("Talon reader does not implement NonBlockingRandomAccessFile");
    return;
  }

  const size_t range_count = expected_.size() / kRangeSize;
  std::vector<arrow::Future<int64_t>> futures;
  futures.reserve(range_count);
  for (auto _ : state) {
    futures.clear();
    for (size_t i = 0; i < range_count; ++i) {
      futures.emplace_back(async_file->ReadAtAsyncInto(
          static_cast<int64_t>(i * kRangeSize), static_cast<int64_t>(kRangeSize), scratch_.data() + i * kRangeSize));
    }

    std::string iteration_error;
    for (auto& future : futures) {
      auto read_result = future.result();
      if (!read_result.ok()) {
        if (iteration_error.empty()) {
          iteration_error = "Talon async range read failed: " + read_result.status().ToString();
        }
      } else if (read_result.ValueOrDie() != static_cast<int64_t>(kRangeSize) && iteration_error.empty()) {
        iteration_error = "Talon async range read returned an unexpected length";
      }
    }
    if (!iteration_error.empty()) {
      state.SkipWithError(iteration_error.c_str());
      break;
    }
    ::benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(expected_.size()));
}

BENCHMARK_DEFINE_F(TalonReaderBenchmark, Concurrent4KiBReadIOPS)(::benchmark::State& state) {
  if (file_ == nullptr) {
    return;
  }
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file_.get());
  if (async_file == nullptr) {
    state.SkipWithError("Talon reader does not implement NonBlockingRandomAccessFile");
    return;
  }

  std::vector<arrow::Future<int64_t>> futures;
  futures.reserve(kIopsBatchSize);
  size_t offset_index = 0;
  int64_t completed_operations = 0;
  for (auto _ : state) {
    futures.clear();
    for (size_t i = 0; i < kIopsBatchSize; ++i) {
      futures.emplace_back(async_file->ReadAtAsyncInto(
          small_io_offsets_[offset_index + i], static_cast<int64_t>(kSmallIoSize), scratch_.data() + i * kSmallIoSize));
    }
    offset_index += kIopsBatchSize;
    if (offset_index == small_io_offsets_.size()) {
      offset_index = 0;
    }

    std::string iteration_error;
    for (auto& future : futures) {
      auto read_result = future.result();
      if (!read_result.ok()) {
        if (iteration_error.empty()) {
          iteration_error = "Talon 4 KiB read failed: " + read_result.status().ToString();
        }
      } else if (read_result.ValueOrDie() != static_cast<int64_t>(kSmallIoSize) && iteration_error.empty()) {
        iteration_error = "Talon 4 KiB read returned an unexpected length";
      }
    }
    if (!iteration_error.empty()) {
      state.SkipWithError(iteration_error.c_str());
      break;
    }
    completed_operations += static_cast<int64_t>(kIopsBatchSize);
    ::benchmark::ClobberMemory();
  }

  // One completed item is one 4 KiB I/O, so items_per_second is the IOPS rate.
  state.SetItemsProcessed(completed_operations);
  state.SetBytesProcessed(completed_operations * static_cast<int64_t>(kSmallIoSize));
}

BENCHMARK_REGISTER_F(TalonReaderBenchmark, FullObjectReadAt)->UseRealTime();
BENCHMARK_REGISTER_F(TalonReaderBenchmark, ConcurrentAsyncRanges)->UseRealTime();
BENCHMARK_REGISTER_F(TalonReaderBenchmark, Concurrent4KiBReadIOPS)->UseRealTime();

}  // namespace milvus_storage
