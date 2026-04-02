// Copyright 2025 Zilliz
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
#include <unordered_map>
#include <vector>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::iceberg {

/// Convert ArrowFileSystemConfig to Iceberg storage options.
/// @throws std::runtime_error for unsupported providers (Tencent, Huawei)
std::unordered_map<std::string, std::string> ToStorageOptions(const ArrowFileSystemConfig& config);

/// Convert a standard-format URI (s3://bucket/key) to Milvus format (s3://endpoint/bucket/key).
/// Returns the original URI unchanged if address is empty or URI is a local path.
std::string ToMilvusUri(const std::string& standard_uri, const std::string& address);

/// Convert delete metadata JSON paths from standard to Milvus-format URIs.
/// Returns the JSON string unchanged if address is empty.
std::string ConvertDeleteMetadataPaths(const std::vector<uint8_t>& json_bytes, const std::string& address);

}  // namespace milvus_storage::iceberg
