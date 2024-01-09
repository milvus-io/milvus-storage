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

#include "file/delete_fragment.h"
#include "storage/manifest.h"
namespace milvus_storage {

namespace internal {
std::unique_ptr<arrow::RecordBatchReader> MakeRecordReader(std::shared_ptr<Manifest> manifest,
                                                           std::shared_ptr<Schema> schema,
                                                           arrow::fs::FileSystem& fs,
                                                           const DeleteFragmentVector& delete_fragments,
                                                           const ReadOptions& options);

bool only_contain_scalar_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

bool only_contain_vector_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

bool filters_only_contain_pk_and_version(std::shared_ptr<Schema> schema, const Filter::FilterSet& filters);

std::unique_ptr<arrow::RecordBatchReader> MakeScanDataReader(std::shared_ptr<Manifest> manifest,
                                                             arrow::fs::FileSystem& fs,
                                                             const ReadOptions& options = ReadOptions());

std::unique_ptr<arrow::RecordBatchReader> MakeScanDeleteReader(std::shared_ptr<Manifest> manifest,
                                                               arrow::fs::FileSystem& fs);
}  // namespace internal
}  // namespace milvus_storage
