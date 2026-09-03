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

#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::iceberg {

/// Convert ArrowFileSystemConfig to credential-free Iceberg reader options.
/// Authentication and credential refresh remain owned by the bound C++ filesystem.
std::unordered_map<std::string, std::string> ToReaderOptions(const ArrowFileSystemConfig& config);

/// Convert ArrowFileSystemConfig to native Iceberg writer options used by tests and benchmarks.
/// Supports AK/SK and IAM credentials; GCP supports IAM only.
arrow::Result<std::unordered_map<std::string, std::string>> ToWriterOptions(const ArrowFileSystemConfig& config);

/// Convert a standard-format URI (s3://bucket/key) to Milvus format (s3://endpoint/bucket/key).
/// Returns the original URI unchanged if address is empty or URI is a local path.
std::string ToMilvusUri(const std::string& standard_uri, const std::string& address);

/// Convert delete metadata JSON paths from standard to Milvus-format URIs.
/// Returns the JSON string unchanged if address is empty.
std::string ConvertDeleteMetadataPaths(const std::vector<uint8_t>& json_bytes, const std::string& address);

/// Strip the @endpoint portion from an Azure Iceberg URI alias.
/// Supports azure, abfs, abfss, wasb, and wasbs while retaining the legacy
/// function name for ABI compatibility. Other schemes and URIs without '@'
/// in their authority are returned unchanged.
std::string StripAbfssEndpoint(const std::string& uri);

/// Normalize any URI to scheme://bucket/path (the Iceberg simple format).
/// Handles both Milvus format (scheme://address/bucket/path → strips address)
/// and Azure recorded format (scheme://container@endpoint/path → strips @endpoint).
std::string MilvusURIToIcebergURI(const std::string& uri);

}  // namespace milvus_storage::iceberg
