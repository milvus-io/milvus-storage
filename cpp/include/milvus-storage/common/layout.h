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

#include "milvus-storage/common/path_util.h"

namespace milvus_storage {

/**
 * @brief Directory Layout Description
 *
 * The storage layout organizes files into separate directories for metadata and data to ensure
 * clean management and separation of concerns.
 *
 * Layout Structure:
 * base_dir/
 * ├── _metadata/                     # Directory for all metadata files
 * │   └── manifest-{version}.avro    # Manifest file containing dataset metadata for a specific version
 * └── _data/                         # Directory for actual data files
 *     └── {group_id}_{uuid}.{format} # Data files (e.g., Parquet, Vortex)
 *
 * Note:
 * - Paths in the manifest are stored relative to the base directory (e.g., "_data/file.parquet").
 */

// Directory names
inline const std::string kMetadataDir = "_metadata";
inline const std::string kDataDir = "_data";

// File names and prefixes
inline const std::string kManifestFileNamePrefix = "manifest-";
inline const std::string kManifestFileNameSuffix = ".avro";

// Full paths relative to base path
inline const std::string kMetadataPath = kMetadataDir + kSep;
inline const std::string kDataPath = kDataDir + kSep;

std::string get_manifest_path(const std::string& base_path);
std::string get_manifest_filename(const size_t& version);
std::string get_manifest_filepath(const std::string& base_path, const size_t& version);

std::string get_data_path(const std::string& base_path);
std::string get_data_filename(const size_t& column_group_id, const std::string& format);
std::string get_data_filepath(const std::string& base_path, const size_t& column_group_id, const std::string& format);
std::string get_data_filepath(const std::string& base_path, const std::string& file_name);

}  // namespace milvus_storage
