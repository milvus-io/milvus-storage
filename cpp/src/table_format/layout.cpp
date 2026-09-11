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

#include "milvus-storage/table_format/layout.h"

#include <charconv>
#include <string>

#include <random>

#include <arrow/filesystem/filesystem.h>
#include <fmt/format.h>

#include "milvus-storage/common/path_util.h"

namespace milvus_storage::api::table_format {

std::string GetCollMetadataDir(const std::string& base_path) {
  return milvus_storage::ConcatenateFilePath(base_path, kCollMetadataDir);
}

std::string GetCollMetadataFilename(int64_t version) { return fmt::format("{}{}", version, kMetadataSuffix); }

std::string GetCollMetadataFilepath(const std::string& base_path, int64_t version) {
  return milvus_storage::ConcatenateFilePath(GetCollMetadataDir(base_path), GetCollMetadataFilename(version));
}

std::string GetCollManifestsDir(const std::string& base_path) {
  return milvus_storage::ConcatenateFilePath(base_path, kCollManifestsDir);
}

std::string GetManifestListFilename(const std::string& uuid) { return fmt::format("{}{}", uuid, kManifestListSuffix); }

std::string GetManifestListFilepath(const std::string& base_path, const std::string& uuid) {
  return milvus_storage::ConcatenateFilePath(GetCollManifestsDir(base_path), GetManifestListFilename(uuid));
}

std::string GetSegmentManifestFilepath(const std::string& base_path, const std::string& uuid) {
  return milvus_storage::ConcatenateFilePath(GetCollManifestsDir(base_path), uuid);
}

std::string GenerateUniqueId() {
  static std::mt19937_64 rng(std::random_device{}());
  return fmt::format("{:016x}", rng());
}

arrow::Result<int64_t> GetLatestMetadataVersion(const milvus_storage::ArrowFileSystemPtr& fs,
                                                const std::string& base_path) {
  std::string metadata_dir = GetCollMetadataDir(base_path);
  ARROW_ASSIGN_OR_RAISE(auto dir_info, fs->GetFileInfo(metadata_dir));
  if (dir_info.type() == arrow::fs::FileType::NotFound) {
    return 0;
  }

  arrow::fs::FileSelector selector;
  selector.base_dir = metadata_dir;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

  int64_t latest_version = 0;
  for (const auto& file_info : file_infos) {
    const std::string file_name = file_info.base_name();
    // Must end with suffix ".metadata.avro"
    if (file_name.size() <= kMetadataSuffix.length()) {
      continue;
    }
    if (file_name.substr(file_name.size() - kMetadataSuffix.length()) != kMetadataSuffix) {
      continue;
    }
    // Extract version number before the suffix
    std::string version_str = file_name.substr(0, file_name.length() - kMetadataSuffix.length());
    int64_t version = 0;
    auto [ptr, ec] = std::from_chars(version_str.data(), version_str.data() + version_str.size(), version);
    if (ec != std::errc() || ptr != version_str.data() + version_str.size()) {
      continue;
    }
    latest_version = std::max(latest_version, version);
  }

  return latest_version;
}

}  // namespace milvus_storage::api::table_format
