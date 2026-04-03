// Copyright 2024 Zilliz
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

#include "milvus-storage/format/format.h"

namespace milvus_storage {

class IcebergFormat final : public Format {
  public:
  [[nodiscard]] arrow::Result<std::vector<api::ColumnGroupFile>> explore(const std::string& explore_dir,
                                                                         const api::Properties& properties) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> create_reader(
      const std::shared_ptr<arrow::Schema>& read_schema,
      const api::ColumnGroupFile& file,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever) override;

  [[nodiscard]] arrow::Result<std::unique_ptr<FormatWriter>> create_writer(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& file_path,
      const std::string& base_path,
      const api::Properties& properties) override;
};

}  // namespace milvus_storage
