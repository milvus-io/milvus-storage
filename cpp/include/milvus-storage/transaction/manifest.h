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

#include <memory>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/common/serializable.h"

namespace milvus_storage::api::transaction {

using Manifest = milvus_storage::api::ColumnGroups;
using ManifestPtr = std::shared_ptr<milvus_storage::api::ColumnGroups>;

static_assert(std::is_base_of_v<Serializable, Manifest>, "Manifest must be Serializable");

}  // namespace milvus_storage::api::transaction