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
static const std::string kMetadataDir = "_metadata";
static const std::string kDataDir = "_data";

// File names and prefixes
static const std::string kManifestFileNamePrefix = "manifest-";
static const std::string kManifestFileNameSuffix = ".avro";

// Full paths relative to base path
static const std::string kMetadataPath = kMetadataDir + "/";
static const std::string kDataPath = kDataDir + "/";

static const std::string kManifestFilePrefix = kMetadataPath + kManifestFileNamePrefix;

static std::string get_manifest_file_name(int64_t version) {
  return kManifestFilePrefix + std::to_string(version) + kManifestFileNameSuffix;
}

}  // namespace milvus_storage
