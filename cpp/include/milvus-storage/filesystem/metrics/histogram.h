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

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

namespace milvus_storage::metrics {

// Fixed-bound histogram. Non-cumulative buckets: an observation v is
// placed in the first bucket i with v <= bounds[i]; values past the last
// bound are counted in `count`/`sum` but no bucket (overflow recoverable
// as count - sum(buckets)). All updates memory_order_relaxed.
class Histogram {
  public:
  Histogram(const int64_t* bounds, int num) : bounds_(bounds), num_(num), buckets_(num) {}

  void Observe(int64_t v) {
    sum_.fetch_add(v, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < num_; ++i) {
      if (v <= bounds_[i]) {
        buckets_[i].fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
  }

  int64_t sum() const { return sum_.load(std::memory_order_relaxed); }
  int64_t count() const { return count_.load(std::memory_order_relaxed); }

  void CopyBuckets(int64_t* out, int num) const {
    for (int i = 0; i < num && i < num_; ++i) {
      out[i] = buckets_[i].load(std::memory_order_relaxed);
    }
  }

  void Reset() {
    sum_.store(0, std::memory_order_relaxed);
    count_.store(0, std::memory_order_relaxed);
    for (auto& b : buckets_) {
      b.store(0, std::memory_order_relaxed);
    }
  }

  private:
  const int64_t* bounds_;
  int num_;
  std::atomic<int64_t> sum_{0};
  std::atomic<int64_t> count_{0};
  std::vector<std::atomic<int64_t>> buckets_;
};

}  // namespace milvus_storage::metrics
