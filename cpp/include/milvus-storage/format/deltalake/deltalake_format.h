// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#pragma once

#include "milvus-storage/format/format.h"

namespace milvus_storage {

class DeltaLakeFormat final : public Format {
  public:
  [[nodiscard]] arrow::Result<std::vector<api::ColumnGroupFile>> explore(
      const std::string& explore_dir, const std::shared_ptr<const api::Properties>& properties) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> create_reader(
      const std::shared_ptr<arrow::Schema>& read_schema,
      const api::ColumnGroupFile& file,
      const std::shared_ptr<const api::Properties>& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever) override;

  [[nodiscard]] arrow::Result<std::unique_ptr<FormatWriter>> create_writer(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::string& file_path,
      const std::string& base_path,
      const std::shared_ptr<const api::Properties>& properties) override;
};

}  // namespace milvus_storage
