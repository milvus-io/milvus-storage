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

#include "milvus-storage/format/format.h"

#include <arrow/filesystem/filesystem.h>
#include <fmt/format.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/parquet/parquet_format.h"
#include "milvus-storage/format/vortex/vortex_format.h"
#include "milvus-storage/format/lance/lance_format.h"
#include "milvus-storage/format/iceberg/iceberg_format.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

arrow::Result<Format*> Format::get(const std::string& format) {
  static ParquetFormat parquet_fmt;
  static VortexFormat vortex_fmt;
  static LanceFormat lance_fmt;
  static IcebergFormat iceberg_fmt;

  if (format == LOON_FORMAT_PARQUET) {
    return &parquet_fmt;
  }
  if (format == LOON_FORMAT_VORTEX) {
    return &vortex_fmt;
  }
  if (format == LOON_FORMAT_LANCE_TABLE) {
    return &lance_fmt;
  }
  if (format == LOON_FORMAT_ICEBERG_TABLE) {
    return &iceberg_fmt;
  }
  return arrow::Status::Invalid(fmt::format("Unknown file format: {}", format));
}

arrow::Result<std::vector<api::ColumnGroupFile>> PlainFormat::explore(const std::string& explore_dir,
                                                                      const api::Properties& properties) {
  // Resolve URI — keep full parsed URI for constructing file URIs later
  ARROW_ASSIGN_OR_RAISE(auto explore_uri, StorageUri::Parse(explore_dir));

  // Get filesystem
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, explore_dir));

  // List files
  arrow::fs::FileSelector selector;
  selector.base_dir = explore_uri.key;
  selector.allow_not_found = false;
  selector.recursive = false;
  selector.max_recursion = 0;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));

  // Build URI base
  StorageUri uri_base;
  if (!explore_uri.scheme.empty()) {
    uri_base.scheme = explore_uri.scheme;
    uri_base.address = explore_uri.address;
    uri_base.bucket_name = explore_uri.bucket_name;
  } else if (IsLocalFileSystem(fs)) {
    uri_base.scheme = fs->type_name();
    uri_base.bucket_name = "local";
  } else {
    uri_base.scheme = fs->type_name();
    ARROW_ASSIGN_OR_RAISE(uri_base.address, api::GetValue<std::string>(properties, PROPERTY_FS_ADDRESS));
    ARROW_ASSIGN_OR_RAISE(uri_base.bucket_name, api::GetValue<std::string>(properties, PROPERTY_FS_BUCKET_NAME));
  }

  std::vector<api::ColumnGroupFile> files;
  for (const auto& file_info : file_infos) {
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    uri_base.key = file_info.path();
    ARROW_ASSIGN_OR_RAISE(auto file_uri, StorageUri::Make(uri_base));
    files.emplace_back(api::ColumnGroupFile{
        std::move(file_uri),
        -1, /*start_index */
        -1, /*end_index */
        {}, /*properties */
    });
  }

  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> PlainFormat::create_reader(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, file.path));
  ARROW_ASSIGN_OR_RAISE(auto uri, StorageUri::Parse(file.path));

  auto reader = make_reader(fs, read_schema, uri.key, properties, needed_columns, key_retriever,
                            file.Get<uint64_t>(api::kPropertyFileSize), file.Get<uint64_t>(api::kPropertyFooterSize));
  ARROW_RETURN_NOT_OK(reader->open());
  return reader;
}

arrow::Result<std::unique_ptr<FormatWriter>> PlainFormat::create_writer(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& file_path,
    const std::string& /*base_path*/,
    const api::Properties& properties) {
  return make_writer(fs, schema, file_path, properties);
}

}  // namespace milvus_storage
