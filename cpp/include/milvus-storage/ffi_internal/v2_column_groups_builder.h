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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "milvus-storage/ffi_c.h"

namespace milvus_storage {

// Allocate a LoonColumnGroups* from caller-provided per-group column names,
// file paths, and per-file row counts. Used by the V2 (non-manifest) read
// path, where the column-group layout is already known from the snapshot
// AVRO + parquet footer kv-metadata and callers want to feed the packed
// reader without resolving a `.milvus_manifest` file.
//
// The returned pointer is owned by the caller and must be released with
// `loon_column_groups_destroy` (matches the allocation scheme used here:
// `new[]` + per-string heap copy).
//
// start_index / end_index per file are computed as cumulative row offsets
// within each column group; the packed reader rejects negative end_index,
// so row counts must be supplied explicitly.
//
// Throws std::invalid_argument on:
//   - mismatched outer lengths between the three vectors
//   - zero column groups
//   - a group with empty columns or empty files
//   - a group where rowCounts.size() != files.size()
LoonColumnGroups* BuildLoonColumnGroups(
    const std::vector<std::vector<std::string>>& columns_per_group,
    const std::vector<std::vector<std::string>>& files_per_group,
    const std::vector<std::vector<int64_t>>& row_counts_per_group);

}  // namespace milvus_storage
