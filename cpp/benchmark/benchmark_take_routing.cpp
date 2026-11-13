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

#include <benchmark/benchmark.h>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/format/async_tasks.h"

namespace milvus_storage::api {
namespace {

void TakeRouting(benchmark::State& state) {
  constexpr int64_t rows_per_file = 2048;
  const int64_t files = state.range(0);
  const int64_t count = state.range(1);
  const bool late = state.range(2) != 0;
  ColumnGroups groups{std::make_shared<ColumnGroup>()};
  groups.front()->files.resize(files, {.start_index = 19, .end_index = 19 + rows_per_file});
  std::vector<int64_t> rows(count);
  for (int64_t i = 0; i < count; ++i) {
    rows[i] = late ? files * rows_per_file - count + i : i * files * rows_per_file / count;
  }

  for (auto _ : state) {
    auto tasks = TakeTask::Build(groups, rows);
    if (!tasks.ok()) {
      state.SkipWithError(tasks.status().ToString().c_str());
      break;
    }
    benchmark::DoNotOptimize(*tasks);
  }
  state.SetLabel(late ? "last file" : "spread");
}

BENCHMARK(TakeRouting)->ArgsProduct({{1, 32, 256}, {1, 128, 1024}, {0, 1}})->ArgNames({"files", "rows", "late"});

}  // namespace
}  // namespace milvus_storage::api
