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

#include <string>
#include <unordered_map>

#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

/// Cloud storage options as key-value pairs, used by Rust bridges (Lance, Iceberg, etc.)
/// to authenticate and access cloud object stores.
using CloudStorageOptions = std::unordered_map<std::string, std::string>;

/// Convert ArrowFileSystemConfig to CloudStorageOptions for cloud storage access.
/// Returns an empty map for local storage.
///
/// Supported cloud providers:
/// - AWS S3: aws_access_key_id, aws_secret_access_key, aws_region, aws_endpoint, allow_http
/// - Azure: azure_storage_account_name, azure_storage_account_key, azure_endpoint, allow_http
/// - GCP: credentials via environment/service account
/// - Aliyun OSS: oss_access_key_id, oss_secret_access_key, oss_region, oss_endpoint
///
/// @throws std::runtime_error for unsupported providers (Tencent, Huawei)
CloudStorageOptions ToCloudStorageOptions(const ArrowFileSystemConfig& config);

}  // namespace milvus_storage
