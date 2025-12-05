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

#include <arrow/filesystem/filesystem.h>
#include <vector>

#include "milvus-storage/column_groups.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::api {

class ColumnGroupLazyReader {
  public:
  virtual ~ColumnGroupLazyReader() = default;

  /**
   * @brief Take a table from the column group
   *
   * @param row_indices the row indices to take, MUST be uniqued and sorted
   * @return arrow::Result<std::shared_ptr<arrow::Table>>
   */
  virtual arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) = 0;

  static arrow::Result<std::unique_ptr<ColumnGroupLazyReader>> create(
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const milvus_storage::api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever);
};

};  // namespace milvus_storage::api