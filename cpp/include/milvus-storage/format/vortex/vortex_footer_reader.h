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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>

namespace milvus_storage::vortex {

struct VortexFieldLayout;

class VortexFooterReader {
  public:
  VortexFooterReader(std::shared_ptr<arrow::fs::FileSystem> sparse_fs,
                     std::string sparse_path,
                     std::string path,
                     uint64_t file_size = 0,
                     uint64_t footer_size = 0);
  ~VortexFooterReader();

  // Opens the required Vortex footer once. When load_zonemap is true, also
  // materializes V1/V2 zonemap segments into the sparse file for predicate
  // pruning. Without zonemap loading, row-group pruning returns all candidates.
  // Do not call Open again; create a new reader for another read mode.
  arrow::Status Open(const std::shared_ptr<arrow::fs::FileSystem>& fs, bool load_zonemap = true);

  bool opened() const;

  uint64_t rows() const;

  std::shared_ptr<arrow::Schema> file_schema() const;

  const std::string& path() const;

  uint64_t file_size() const;

  uint64_t footer_size() const;

  // Vortex physical layout metadata used by local-format mapping.
  arrow::Result<VortexFieldLayout> GetFieldLayout(const std::string& field_name) const;

  arrow::Result<std::vector<uint64_t>> PruneRowGroups(const std::string& predicate,
                                                      const std::vector<uint64_t>& candidate_row_group_ids) const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace milvus_storage::vortex
