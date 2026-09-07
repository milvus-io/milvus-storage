// Copyright 2025 Zilliz
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

// Throughput benchmarks for the Talon-backed input file. They read a single
// object served by a running Talon deployment (a local coordinator + worker in
// dev, whatever CI points them at otherwise), so they only build with WITH_TALON
// and only run when a coordinator + object URI are supplied via the environment;
// otherwise each case reports itself skipped, mirroring the integration test.
//
//   TALON_COORDINATOR_ADDR - control address of a running coordinator (required)
//   TALON_TEST_URI         - an object URI the deployment can serve (required)
//   TALON_BENCH_BLOCK_SIZE - async block size in bytes (optional, default 4 MiB)
#ifdef WITH_TALON

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <arrow/buffer.h>
#include <arrow/util/future.h>

#include "milvus-storage/filesystem/talon/talon_client.h"
#include "milvus-storage/filesystem/talon/talon_input_file.h"

namespace milvus_storage::talon {
namespace {

constexpr int64_t kDefaultBlockSize = 4 * 1024 * 1024;

struct TalonBenchEnv {
  std::string coordinator;
  std::string uri;
  int64_t block_size = kDefaultBlockSize;
  bool available = false;
};

TalonBenchEnv ReadEnv() {
  TalonBenchEnv env;
  const char* addr = std::getenv("TALON_COORDINATOR_ADDR");
  const char* uri = std::getenv("TALON_TEST_URI");
  if (addr != nullptr && addr[0] != '\0' && uri != nullptr && uri[0] != '\0') {
    env.coordinator = addr;
    env.uri = uri;
    env.available = true;
  }
  const char* block = std::getenv("TALON_BENCH_BLOCK_SIZE");
  if (block != nullptr && block[0] != '\0') {
    const int64_t parsed = std::atoll(block);
    if (parsed > 0) {
      env.block_size = parsed;
    }
  }
  return env;
}

// Resolves a client and object once, then reads the object into the Talon cache
// so the measured loop reports steady-state (cache-hit) throughput through the
// worker rather than the one-off origin fetch.
class TalonReaderBenchmark : public ::benchmark::Fixture {
  protected:
  void SetUp(::benchmark::State& state) override {
    env_ = ReadEnv();
    if (!env_.available) {
      state.SkipWithError("set TALON_COORDINATOR_ADDR and TALON_TEST_URI to run Talon benchmarks");
      return;
    }

    auto client = TalonClient::Make(env_.coordinator);
    if (!client.ok()) {
      state.SkipWithError(client.status().ToString().c_str());
      return;
    }
    client_ = client.ValueOrDie();

    file_ = std::make_shared<TalonInputFile>(client_, env_.uri);
    auto size = file_->GetSize();
    if (!size.ok()) {
      state.SkipWithError(size.status().ToString().c_str());
      return;
    }
    size_ = size.ValueOrDie();
    if (size_ <= 0) {
      state.SkipWithError("TALON_TEST_URI resolves to an empty object");
      return;
    }
    scratch_.assign(static_cast<size_t>(size_), 0);

    // Warm the cache so the first measured iteration is not the origin fetch.
    auto warm = file_->ReadAt(0, size_);
    if (!warm.ok()) {
      state.SkipWithError(warm.status().ToString().c_str());
      return;
    }
  }

  void TearDown(::benchmark::State& /*state*/) override {
    if (file_ != nullptr && !file_->closed()) {
      auto status = file_->Close();
      if (!status.ok()) {
        // Best-effort cleanup; nothing actionable during benchmark teardown.
      }
    }
    file_.reset();
    client_.reset();
    scratch_.clear();
    scratch_.shrink_to_fit();
  }

  TalonBenchEnv env_;
  std::shared_ptr<TalonClient> client_;
  std::shared_ptr<TalonInputFile> file_;
  int64_t size_ = 0;
  std::vector<uint8_t> scratch_;
};

// Full-object read through the buffer-returning path (what a format reader that
// pulls a whole footer/column chunk does).
BENCHMARK_DEFINE_F(TalonReaderBenchmark, FullObjectReadAt)(::benchmark::State& state) {
  if (file_ == nullptr) {
    return;  // SetUp already flagged the skip.
  }
  for (auto _ : state) {
    auto buffer = file_->ReadAt(0, size_);
    if (!buffer.ok()) {
      state.SkipWithError(buffer.status().ToString().c_str());
      break;
    }
    ::benchmark::DoNotOptimize(buffer.ValueUnsafe()->data());
  }
  state.SetBytesProcessed(state.iterations() * size_);
}

// Whole object read as a fan-out of fixed-size async range reads into a
// caller-owned buffer — the NonBlockingReadAtFile fast path the Parquet reader
// uses. All reads for one iteration are issued before any is awaited, so this
// measures Talon's concurrent block fetch, not serialized round trips.
BENCHMARK_DEFINE_F(TalonReaderBenchmark, ConcurrentAsyncBlocks)(::benchmark::State& state) {
  if (file_ == nullptr) {
    return;  // SetUp already flagged the skip.
  }
  const int64_t block = env_.block_size;
  std::vector<arrow::Future<int64_t>> futures;
  futures.reserve(static_cast<size_t>((size_ + block - 1) / block));

  for (auto _ : state) {
    futures.clear();
    for (int64_t offset = 0; offset < size_; offset += block) {
      const int64_t nbytes = std::min(block, size_ - offset);
      futures.push_back(file_->ReadAtAsyncInto(offset, nbytes, scratch_.data() + offset));
    }
    bool failed = false;
    for (auto& future : futures) {
      auto read = future.result();
      if (!read.ok()) {
        state.SkipWithError(read.status().ToString().c_str());
        failed = true;
        break;
      }
    }
    if (failed) {
      break;
    }
  }
  state.SetBytesProcessed(state.iterations() * size_);
}

BENCHMARK_REGISTER_F(TalonReaderBenchmark, FullObjectReadAt)->Unit(::benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(TalonReaderBenchmark, ConcurrentAsyncBlocks)->Unit(::benchmark::kMillisecond)->UseRealTime();

}  // namespace
}  // namespace milvus_storage::talon

#endif  // WITH_TALON
