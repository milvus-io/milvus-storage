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

#include "milvus-storage/ffi_internal/v2_column_groups_builder.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "milvus-storage/common/config.h"

namespace milvus_storage {

namespace {

// Null-terminated copy of `s` that `delete[]` can reclaim — matches the
// expectations of `loon_column_groups_destroy`.
char* dup_cstr(const std::string& s) {
  char* p = new char[s.size() + 1];
  std::memcpy(p, s.data(), s.size());
  p[s.size()] = '\0';
  return p;
}

}  // namespace

LoonColumnGroups* BuildLoonColumnGroups(const std::vector<std::vector<std::string>>& columns_per_group,
                                        const std::vector<std::vector<std::string>>& files_per_group,
                                        const std::vector<std::vector<int64_t>>& row_counts_per_group) {
  const size_t num_groups = columns_per_group.size();
  if (num_groups != files_per_group.size() || num_groups != row_counts_per_group.size()) {
    throw std::invalid_argument("per-group array length mismatch: cols=" + std::to_string(num_groups) +
                                ", files=" + std::to_string(files_per_group.size()) +
                                ", rowCounts=" + std::to_string(row_counts_per_group.size()));
  }
  if (num_groups == 0) {
    throw std::invalid_argument("at least one column group is required");
  }

  LoonColumnGroups* cgroups = new LoonColumnGroups{};
  try {
    cgroups->num_of_column_groups = static_cast<uint32_t>(num_groups);
    cgroups->column_group_array = new LoonColumnGroup[num_groups]{};

    for (size_t g = 0; g < num_groups; ++g) {
      const auto& cols = columns_per_group[g];
      const auto& files = files_per_group[g];
      const auto& rcs = row_counts_per_group[g];
      if (cols.empty() || files.empty()) {
        throw std::invalid_argument("group[" + std::to_string(g) + "]: columns/files must be non-empty");
      }
      if (rcs.size() != files.size()) {
        throw std::invalid_argument("group[" + std::to_string(g) + "]: rowCounts.length (" +
                                    std::to_string(rcs.size()) + ") != files.length (" + std::to_string(files.size()) +
                                    ")");
      }

      LoonColumnGroup& out = cgroups->column_group_array[g];
      out.num_of_columns = static_cast<uint32_t>(cols.size());
      out.columns = new const char* [cols.size()] {};
      out.format = dup_cstr(LOON_FORMAT_PARQUET);
      out.num_of_files = static_cast<uint32_t>(files.size());
      out.files = new LoonColumnGroupFile[files.size()]{};

      for (size_t c = 0; c < cols.size(); ++c) {
        out.columns[c] = dup_cstr(cols[c]);
      }

      // Cumulative row offsets within the group: file i covers
      // [start_i, end_i) where start_i = sum(rowCounts[0..i)).
      int64_t cumulative = 0;
      for (size_t f = 0; f < files.size(); ++f) {
        out.files[f].path = dup_cstr(files[f]);
        out.files[f].start_index = cumulative;
        out.files[f].end_index = cumulative + rcs[f];
        cumulative += rcs[f];
        out.files[f].property_keys = nullptr;
        out.files[f].property_values = nullptr;
        out.files[f].num_properties = 0;
      }
    }
    return cgroups;
  } catch (...) {
    loon_column_groups_destroy(cgroups);
    throw;
  }
}

}  // namespace milvus_storage
