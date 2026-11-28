// Copyright 2026 Zilliz
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
#include "rust/cxx.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

namespace milvus_storage::iceberg::ffi {
struct SourceUriResult;
SourceUriResult resolve_source_uri(const FileSystemWrapper& filesystem, rust::Str uri) noexcept;
}  // namespace milvus_storage::iceberg::ffi
