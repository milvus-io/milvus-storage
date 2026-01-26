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
// Export function allocates and returns the structure - caller must call loon_manifest_destroy to free
arrow::Status manifest_export(const std::shared_ptr<milvus_storage::api::Manifest>& manifest,
                              LoonManifest** out_cmanifest);

arrow::Status manifest_import(const LoonManifest* cmanifest,
                              std::shared_ptr<milvus_storage::api::Manifest>* out_manifest);

// Get debug string representation of manifest
std::string manifest_debug_string(const LoonManifest* cmanifest);

// Helper functions for column groups only (for backward compatibility)
// Export function allocates and returns the structure - caller must call `loon_column_groups_destroy` to free
arrow::Status column_groups_export(const milvus_storage::api::ColumnGroups& cgs, LoonColumnGroups** out_ccgs);

arrow::Status column_groups_import(const LoonColumnGroups* ccgs, milvus_storage::api::ColumnGroups* out_cgs);

// Get debug string representation of column groups
std::string column_groups_debug_string(const LoonColumnGroups* ccgs);

}  // namespace milvus_storage