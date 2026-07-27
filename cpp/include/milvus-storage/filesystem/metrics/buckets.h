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

#include <array>
#include <cstdint>

namespace milvus_storage::metrics {

// Latency bucket upper bounds, microseconds. 0.1ms .. 130s.
inline constexpr std::array<int64_t, 16> kLatencyBoundsUs = {100,     250,     500,      1000,     2500,   5000,
                                                             10000,   25000,   50000,    100000,   250000, 500000,
                                                             1000000, 5000000, 30000000, 130000000};

// Size bucket upper bounds, bytes. 256B .. 16GB.
inline constexpr std::array<int64_t, 14> kSizeBoundsBytes = {256,       1024,       4096,       16384,      65536,
                                                             262144,    1048576,    4194304,    16777216,   67108864,
                                                             268435456, 1073741824, 4294967296, 17179869184};

}  // namespace milvus_storage::metrics
