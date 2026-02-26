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

#include "milvus-storage/common/layout.h"

#include <string>
#include <filesystem>

#include <fmt/core.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/path_util.h"

namespace milvus_storage {

std::string get_manifest_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kMetadataPath).lexically_normal().string();
}

std::string get_manifest_filename(const size_t& version) {
  return fmt::format("{}{}{}", kManifestFileNamePrefix, version, kManifestFileNameSuffix);
}

std::string get_manifest_filepath(const std::string& base_path, const size_t& version) {
  std::filesystem::path manifest_path(get_manifest_path(base_path));
  return (manifest_path / get_manifest_filename(version)).lexically_normal().string();
}

std::string get_data_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kDataPath).lexically_normal().string();
}

std::string get_data_filename(const size_t& column_group_id, const std::string& format) {
  static boost::uuids::random_generator random_gen;
  boost::uuids::uuid random_uuid = random_gen();
  const std::string uuid_str = boost::uuids::to_string(random_uuid);
  // named as {group_id}_{uuid}.{format}
  return fmt::format("{}_{}.{}", column_group_id, uuid_str, format);
}

std::string get_data_filepath(const std::string& base_path, const size_t& column_group_id, const std::string& format) {
  std::filesystem::path data_path(get_data_path(base_path));
  return (data_path / get_data_filename(column_group_id, format)).lexically_normal().string();
}

std::string get_data_filepath(const std::string& base_path, const std::string& file_name) {
  std::filesystem::path data_path(get_data_path(base_path));
  return (data_path / file_name).lexically_normal().string();
}

std::string get_delta_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kDeltaPath).lexically_normal().string();
}

std::string get_delta_filepath(const std::string& base_path, const std::string& file_name) {
  std::filesystem::path delta_path(get_delta_path(base_path));
  return (delta_path / file_name).lexically_normal().string();
}

std::string get_stats_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kStatsPath).lexically_normal().string();
}

std::string get_stats_filepath(const std::string& base_path, const std::string& file_name) {
  std::filesystem::path stats_path(get_stats_path(base_path));
  return (stats_path / file_name).lexically_normal().string();
}

std::string get_index_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kIndexPath).lexically_normal().string();
}

std::string get_index_filepath(const std::string& base_path, const std::string& file_name) {
  std::filesystem::path index_path(get_index_path(base_path));
  return (index_path / file_name).lexically_normal().string();
}

}  // namespace milvus_storage
