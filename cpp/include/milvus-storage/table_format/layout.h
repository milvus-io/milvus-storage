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

#include <cstdint>
#include <string>

#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::api::table_format {

// Collection-level directory names
inline const std::string kCollMetadataDir = "_metadata";
inline const std::string kCollManifestsDir = "_manifests";
inline const std::string kCollIndexDir = "_index";
inline const std::string kCollDataDir = "_data";

// File naming
inline const std::string kMetadataSuffix = ".metadata.avro";
inline const std::string kManifestListSuffix = ".avro";

// Path builders
std::string GetCollMetadataDir(const std::string& base_path);
std::string GetCollMetadataFilename(int64_t version);
std::string GetCollMetadataFilepath(const std::string& base_path, int64_t version);

std::string GetCollManifestsDir(const std::string& base_path);
std::string GetManifestListFilename(const std::string& uuid);
std::string GetManifestListFilepath(const std::string& base_path, const std::string& uuid);
std::string GetSegmentManifestFilepath(const std::string& base_path, const std::string& uuid);

// Unique ID generation (16-char random hex, 64 bits of entropy)
std::string GenerateUniqueId();

// Version discovery: scan metadata/ dir, return highest version number.
// Returns 0 if no metadata files exist.
arrow::Result<int64_t> GetLatestMetadataVersion(const milvus_storage::ArrowFileSystemPtr& fs,
                                                const std::string& base_path);

}  // namespace milvus_storage::api::table_format
