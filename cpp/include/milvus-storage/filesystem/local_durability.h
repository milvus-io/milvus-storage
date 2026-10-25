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

// Persist the directory entry and all ancestor links. Existence is not proof
// of durability: an existing parent can belong to another in-flight writer.
arrow::Status SyncLocalDirectoryTree(const std::string& directory);

// Complete-object writes publish a synchronized sibling temporary file only on
// explicit successful Close. exclusive publication never replaces an existing
// object (also across processes). Append streams sync their existing file on
// Close. Write/Flush do not issue a disk synchronization per small input chunk.
arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenDurableLocalOutput(const std::string& path,
                                                                               bool exclusive = false,
                                                                               bool append = false);

}  // namespace milvus_storage
