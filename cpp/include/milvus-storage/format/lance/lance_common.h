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

#include <string>
#include <utility>

#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"
#include "lance_bridge.h"

namespace milvus_storage::lance {

/// Convert ArrowFileSystemConfig to Lance storage options.
/// Key format: aws_access_key_id, aws_secret_access_key, aws_region, aws_endpoint, etc.
/// @throws std::runtime_error for unsupported providers (Tencent, Huawei)
StorageOptions ToStorageOptions(const ArrowFileSystemConfig& config);

/// Parse a Lance URI to extract the base path and fragment ID.
/// URI format: {base_path}?fragment_id={fragment_id}
///
/// @param uri The full URI containing base path and fragment ID
/// @return A pair of (base_path, fragment_id) or an error if the format is invalid
arrow::Result<std::pair<std::string, uint64_t>> ParseLanceUri(const std::string& uri);

/// Construct a Lance URI from a base path and fragment ID.
///
/// @param base_path The base path (e.g., "s3://bucket/path" or "/tmp/path")
/// @param fragment_id The fragment ID to append
/// @return The constructed URI in format: {base_path}?fragment_id={fragment_id}
std::string MakeLanceUri(const std::string& base_path, uint64_t fragment_id);

/// Build a Lance-compatible base URI from ArrowFileSystemConfig and a relative path.
///
/// For cloud storage, constructs URIs like:
/// - AWS S3: s3://bucket/path
/// - Azure: az://container/path
/// - GCP: gs://bucket/path
/// - Aliyun OSS: oss://bucket/path
///
/// For local storage, returns the absolute path.
///
/// Returns error for unsupported providers (Tencent, Huawei).
///
/// @param config The filesystem configuration containing storage type and bucket info
/// @param relative_path The relative path within the bucket/filesystem
/// @return The constructed base URI or an error
arrow::Result<std::string> BuildLanceBaseUri(const ArrowFileSystemConfig& config, const std::string& relative_path);

}  // namespace milvus_storage::lance
