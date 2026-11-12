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

#include <cstdint>
#include <memory>
#include <string>

#include <arrow/result.h>

namespace arrow::io {
class RandomAccessFile;
}

namespace arrow::fs {
class FileSystem;
}

namespace milvus_storage {
struct ArrowFileSystemConfig;

enum class TalonMode : uint8_t { Disabled, Full, SmallReads };
}  // namespace milvus_storage

namespace milvus_storage::talon::internal {

/// Update the existing file's routing flag for subsequent content reads.
///
/// For an open-only Talon file, enable=true restores its configured Talon mode
/// (Full or SmallReads), and enable=false routes reads through its origin handle.
/// Talon files with open_only=false always retain their configured Talon mode.
///
/// The return value is the capability used for FFI read dispatch. Talon-enabled
/// files return true. When Talon is disabled, it reports whether the origin
/// implements NonBlockingRandomAccessFile. Ordinary files report their own
/// NonBlockingRandomAccessFile capability.
///
/// The caller keeps the file open and alive while controlling and reading it.
/// Each read captures the routing flag when it starts.
bool ControlTalonForFile(arrow::io::RandomAccessFile* file, bool enable);

/// Creates a Talon read decorator around `origin_fs`.
///
/// No concrete filesystem type is required. `bucket` is the normalized,
/// non-empty bucket name used with the cloud provider and object key to
/// construct Talon's ObjectId. Reads use Talon and fall back to `origin_fs` if
/// stat or read fails. Talon reader construction errors propagate directly.
/// All other operations delegate to `origin_fs`. A failed read retries its
/// entire range through the origin.
arrow::Result<std::shared_ptr<arrow::fs::FileSystem>> MakeTalonFileSystem(
    const ArrowFileSystemConfig& config, std::shared_ptr<arrow::fs::FileSystem> origin_fs, std::string bucket);

}  // namespace milvus_storage::talon::internal
