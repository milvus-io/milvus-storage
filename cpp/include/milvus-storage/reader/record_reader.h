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
#include "reader/filter_query_record_reader.h"
namespace milvus_storage {

struct RecordReader {
  static std::unique_ptr<arrow::RecordBatchReader> MakeRecordReader(std::shared_ptr<Manifest> manifest,
                                                                    std::shared_ptr<Schema> schema,
                                                                    std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                    DeleteFragmentVector delete_fragments,
                                                                    std::shared_ptr<ReadOptions>& options);

  static bool only_contain_scalar_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

  static bool only_contain_vector_columns(std::shared_ptr<Schema> schema, const std::set<std::string>& related_columns);

  static bool filters_only_contain_pk_and_version(std::shared_ptr<Schema> schema,
                                                  const std::vector<std::unique_ptr<Filter>>& filters);

  static Result<std::shared_ptr<arrow::RecordBatchReader>> MakeScanDataReader(
      std::shared_ptr<Manifest> manifest, std::shared_ptr<arrow::fs::FileSystem> fs);

  static std::shared_ptr<arrow::RecordBatchReader> MakeScanDeleteReader(std::shared_ptr<Manifest> manifest,
                                                                        std::shared_ptr<arrow::fs::FileSystem> fs);
};
}  // namespace milvus_storage
