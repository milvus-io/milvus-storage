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

#include <arrow/status.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/column_groups.h"

namespace milvus_storage {

// Forward declarations
namespace api {
class Manifest;
struct ColumnGroup;
}  // namespace api

// Main functions for exporting/importing Manifest (includes column groups, delta logs, and stats)
arrow::Status export_manifest(const std::shared_ptr<milvus_storage::api::Manifest>& manifest, CManifest* out_cmanifest);

arrow::Status import_manifest(const CManifest* cmanifest, std::shared_ptr<milvus_storage::api::Manifest>* out_manifest);

// Helper functions for column groups only (for backward compatibility)
arrow::Status export_column_groups(const milvus_storage::api::ColumnGroups& cgs, CColumnGroups* out_ccgs);

arrow::Status import_column_groups(const CColumnGroups* ccgs, milvus_storage::api::ColumnGroups* out_cgs);

}  // namespace milvus_storage