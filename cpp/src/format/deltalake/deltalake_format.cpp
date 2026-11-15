// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "milvus-storage/format/deltalake/deltalake_format.h"

#include <nlohmann/json.hpp>

#include "deltalake/deltalake_bridge.h"
#include "format/deltalake/deltalake_common.h"
#include "milvus-storage/format/deltalake/deltalake_format_reader.h"

namespace milvus_storage {
arrow::Result<std::vector<api::ColumnGroupFile>> DeltaLakeFormat::explore(
    const std::string& explore_dir, const std::shared_ptr<const api::Properties>& properties) {
  ARROW_ASSIGN_OR_RAISE(auto location, deltalake::ResolveLocation(explore_dir, properties));
  ARROW_ASSIGN_OR_RAISE(auto version, api::GetValue<int64_t>(properties, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID));
  ARROW_ASSIGN_OR_RAISE(auto snapshot, deltalake::BlockingDeltaSnapshot::Open(location.url, version,
                                                                              location.filesystem, location.binding));
  ARROW_ASSIGN_OR_RAISE(auto table_uri, deltalake::ToExternalUri(location.url, location.config));
  const auto descriptor =
      nlohmann::json{{"version", 1}, {"table_uri", table_uri}, {"snapshot_version", snapshot->Version()}}.dump();
  std::vector<api::ColumnGroupFile> files;
  for (const auto& info : snapshot->Files()) {
    ARROW_ASSIGN_OR_RAISE(auto path, deltalake::ToExternalUri(info.path, location.config));
    api::ColumnGroupFile file{std::move(path), 0, static_cast<int64_t>(info.row_count), {}};
    file.Set(api::kPropertyFileSize, info.file_size);
    file.Set(api::kPropertyMetadata, descriptor);
    files.push_back(std::move(file));
  }
  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> DeltaLakeFormat::create_reader(
    const std::shared_ptr<arrow::Schema>& schema,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<const api::Properties>& properties,
    const std::vector<std::string>& needed,
    const std::function<std::string(const std::string&)>& key_retriever) {
  using Reader = deltalake::DeltaLakeFormatReader;
  ARROW_ASSIGN_OR_RAISE(auto metadata, Reader::MetaTrait::load_metadata(file, properties, key_retriever));
  return Reader::MetaTrait::create_from_metadata(std::move(metadata), file, schema, needed, "");
}

arrow::Result<std::unique_ptr<FormatWriter>> DeltaLakeFormat::create_writer(
    const std::shared_ptr<arrow::fs::FileSystem>&,
    const std::shared_ptr<arrow::Schema>&,
    const std::string&,
    const std::string&,
    const std::shared_ptr<const api::Properties>&) {
  return arrow::Status::NotImplemented("Delta Lake format does not support writing");
}

}  // namespace milvus_storage
