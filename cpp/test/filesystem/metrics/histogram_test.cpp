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

#include <gtest/gtest.h>
#include <vector>

#include "milvus-storage/filesystem/metrics/buckets.h"
#include "milvus-storage/filesystem/metrics/histogram.h"

namespace milvus_storage::test {

using metrics::Histogram;
using metrics::kLatencyBoundsUs;

TEST(HistogramTest, ObserveRecordsSumCountAndBucket) {
  Histogram h(kLatencyBoundsUs.data(), static_cast<int>(kLatencyBoundsUs.size()));
  h.Observe(100);            // <= bounds[0] (100) -> bucket 0
  h.Observe(1000);           // -> bucket 3 (bound 1000)
  h.Observe(1'000'000'000);  // overflow: past last bound (130s)

  EXPECT_EQ(h.count(), 3);
  EXPECT_EQ(h.sum(), 100 + 1000 + 1'000'000'000);

  std::vector<int64_t> b(kLatencyBoundsUs.size(), 0);
  h.CopyBuckets(b.data(), static_cast<int>(b.size()));
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[3], 1);
  int64_t bucketed = 0;
  for (auto c : b) bucketed += c;
  EXPECT_EQ(bucketed, 2);              // overflow not in any bucket
  EXPECT_EQ(h.count() - bucketed, 1);  // overflow recoverable
}

TEST(HistogramTest, ResetZeroesEverything) {
  Histogram h(kLatencyBoundsUs.data(), static_cast<int>(kLatencyBoundsUs.size()));
  h.Observe(5000);
  h.Reset();
  EXPECT_EQ(h.count(), 0);
  EXPECT_EQ(h.sum(), 0);
  std::vector<int64_t> b(kLatencyBoundsUs.size(), 0);
  h.CopyBuckets(b.data(), static_cast<int>(b.size()));
  for (auto c : b) EXPECT_EQ(c, 0);
}

}  // namespace milvus_storage::test
