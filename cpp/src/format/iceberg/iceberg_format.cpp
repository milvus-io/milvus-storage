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

#include "milvus-storage/format/iceberg/iceberg_format.h"

#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/iceberg/iceberg_common.h"
#include "milvus-storage/filesystem/fs.h"
#include "iceberg_bridge.h"

namespace milvus_storage {

arrow::Result<std::vector<api::ColumnGroupFile>> IcebergFormat::explore(const std::string& explore_dir,
                                                                        const api::Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, explore_dir.c_str()));
  auto storage_options = iceberg::ToStorageOptions(fs_config);

  ARROW_ASSIGN_OR_RAISE(auto snapshot_str, api::GetValue<std::string>(properties, PROPERTY_ICEBERG_SNAPSHOT_ID));
  int64_t snapshot_id = std::stoll(snapshot_str);

  // Convert Milvus URI (scheme://address/bucket/path) to scheme://bucket/path.
  // For S3 this is the final format; for Azure ABFSS, the Rust bridge further
  // expands it to container@account.dfs.endpoint format that opendal requires.
  ARROW_ASSIGN_OR_RAISE(auto parsed_uri, StorageUri::Parse(explore_dir));
  ARROW_ASSIGN_OR_RAISE(auto iceberg_uri, StorageUri::Make(parsed_uri, false));

  auto file_infos = iceberg::PlanFiles(iceberg_uri, snapshot_id, storage_options);

  std::vector<api::ColumnGroupFile> files;
  files.reserve(file_infos.size());
  for (const auto& info : file_infos) {
    // Convert data file path from standard format back to Milvus format
    auto milvus_path = iceberg::ToMilvusUri(info.data_file_path, fs_config.address);

    api::ColumnGroupFile cgf{
        std::move(milvus_path), 0, static_cast<int64_t>(info.record_count - info.num_deleted_rows), {}};
    if (!info.delete_metadata_json.empty()) {
      cgf.Set(api::kPropertyMetadata,
              iceberg::ConvertDeleteMetadataPaths(info.delete_metadata_json, fs_config.address));
    }
    files.emplace_back(std::move(cgf));
  }

  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> IcebergFormat::create_reader(
    const std::shared_ptr<arrow::Schema>& /*read_schema*/,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, file.path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(file.path));

  // Extract delete metadata from properties and convert to vector<uint8_t>
  std::vector<uint8_t> delete_metadata;
  auto meta_str = file.Get<std::string>(api::kPropertyMetadata);
  if (!meta_str.empty()) {
    delete_metadata.assign(meta_str.begin(), meta_str.end());
  }

  auto reader = std::make_shared<iceberg::IcebergFormatReader>(fs, uri.key, file.path, delete_metadata, properties,
                                                               needed_columns, key_retriever);
  ARROW_RETURN_NOT_OK(reader->open());
  return reader;
}

arrow::Result<std::unique_ptr<FormatWriter>> IcebergFormat::create_writer(
    const std::shared_ptr<arrow::fs::FileSystem>& /*fs*/,
    const std::shared_ptr<arrow::Schema>& /*schema*/,
    const std::string& /*file_path*/,
    const std::string& /*base_path*/,
    const api::Properties& /*properties*/) {
  return arrow::Status::NotImplemented("Iceberg format does not support writing");
}

}  // namespace milvus_storage
