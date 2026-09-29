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

#include <string>
#include <vector>

#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/table_format/types.h"

namespace milvus_storage::api::table_format {

class ManifestList final {
  public:
  ManifestList() = default;
  explicit ManifestList(std::vector<ManifestListEntry> entries);

  ManifestList(ManifestList&&) = default;
  ManifestList& operator=(ManifestList&&) = default;
  ~ManifestList() = default;

  [[nodiscard]] arrow::Status serialize(std::ostream& output_stream) const;
  arrow::Status deserialize(std::istream& input_stream);

  [[nodiscard]] std::vector<ManifestListEntry>& entries() { return entries_; }
  [[nodiscard]] const std::vector<ManifestListEntry>& entries() const { return entries_; }

  private:
  ManifestList(const ManifestList&) = delete;
  ManifestList& operator=(const ManifestList&) = delete;

  std::vector<ManifestListEntry> entries_;
};

// Filesystem I/O helpers for manifest list files.
arrow::Result<ManifestList> ReadManifestListFromFile(const milvus_storage::ArrowFileSystemPtr& fs,
                                                     const std::string& path);
arrow::Result<std::string> WriteManifestListToFile(const milvus_storage::ArrowFileSystemPtr& fs,
                                                   const std::string& base_path,
                                                   const ManifestList& ml);

}  // namespace milvus_storage::api::table_format
