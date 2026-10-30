// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <memory>
#include <string>

#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace milvus_storage {

// Synchronize an existing local file, including errors reported by close.
arrow::Status SyncLocalFile(const std::string& path);

// Sync physical directories bottom-up through the provisioned storage root.
// The root and symlinks leading to it must be stable, durably provisioned by
// the caller. Do not delete/recreate that boundary while the filesystem is used.
arrow::Status SyncLocalDirectoryTree(const std::string& directory, const std::string& root);

// Resolve an existing provisioned root, or create and sync a missing root
// through the nearest existing (provisioned) ancestor during initialization.
arrow::Result<std::string> PrepareLocalRoot(const std::string& root);

// Write directly to the target path, synchronizing its file and directory on
// successful Close. Write/Flush do not synchronize each small input chunk.
// Failed or abandoned writes can leave incomplete files, as before.
arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenDurableLocalOutput(const std::string& path,
                                                                               bool append,
                                                                               const std::string& root);

}  // namespace milvus_storage
