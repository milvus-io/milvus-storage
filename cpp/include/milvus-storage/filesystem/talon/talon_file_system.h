// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include <arrow/result.h>

namespace arrow::fs {
class FileSystem;
}

namespace milvus_storage {
struct ArrowFileSystemConfig;
}

namespace milvus_storage::talon::internal {

/// Creates a Talon read decorator around `origin_fs`.
///
/// No concrete filesystem type is required. `bucket` is the normalized,
/// non-empty bucket name used with the cloud provider and object key to
/// construct Talon's ObjectId. Reads use Talon; all other operations delegate
/// to `origin_fs`.
arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> MakeTalonFileSystem(
    const ArrowFileSystemConfig& config, std::shared_ptr<arrow::fs::FileSystem> origin_fs, std::string bucket);

}  // namespace milvus_storage::talon::internal
