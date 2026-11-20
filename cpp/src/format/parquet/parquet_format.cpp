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

#include "milvus-storage/format/parquet/parquet_format.h"

#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/source_format_reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/azure/source_sas_token_policy.h"

namespace milvus_storage {

arrow::Result<std::vector<api::ColumnGroupFile>> ParquetFormat::explore_with_context(
    const std::string& explore_dir,
    const std::shared_ptr<const api::Properties>& properties,
    SourceReadContextPtr context) {
  ARROW_ASSIGN_OR_RAISE(auto source, FilesystemCache::getInstance().get_with_context(properties, explore_dir, context));
  auto resource = AzureSourceResource::Parse(source.source_uri, context->account_name());
  if (!resource)
    return fs::SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_DENIED);
  auto binding = AzureSourceStorageBinding::Create(resource->account, resource->endpoint_suffix, resource->container);
  if (!binding)
    return fs::SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_DENIED);
  arrow::fs::FileSelector selector;
  selector.base_dir = source.path;
  selector.recursive = false;
  ARROW_ASSIGN_OR_RAISE(auto infos, source.filesystem->GetFileInfo(selector));
  ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->CheckState()));
  std::vector<api::ColumnGroupFile> files;
  for (const auto& info : infos) {
    if (info.type() != arrow::fs::FileType::File)
      continue;
    auto file = binding->ResolvePath(info.path());
    if (!file || !resource->Contains(*file))
      return fs::SourceCredentialArrowStatus(LOON_SOURCE_CREDENTIAL_DENIED);
    auto uri = file->CanonicalUri();
    ARROW_RETURN_NOT_OK(fs::SourceCredentialArrowStatus(context->Resolve(uri).status));
    files.emplace_back(api::ColumnGroupFile{std::move(uri), -1, -1, {}});
  }
  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> ParquetFormat::create_reader_with_context(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<const api::Properties>& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
    SourceReadContextPtr context) {
  ARROW_ASSIGN_OR_RAISE(auto source, FilesystemCache::getInstance().get_with_context(properties, file.path, context));
  auto reader = make_reader(source.filesystem, read_schema, source.path, properties, needed_columns, key_retriever,
                            file.Get<uint64_t>(api::kPropertyFileSize), file.Get<uint64_t>(api::kPropertyFooterSize));
  auto guarded = GuardSourceFormatReader(std::move(reader), std::move(context));
  ARROW_RETURN_NOT_OK(guarded->open());
  return guarded;
}

std::shared_ptr<FormatReader> ParquetFormat::make_reader(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& /*read_schema*/,
    const std::string& resolved_path,
    const std::shared_ptr<const api::Properties>& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
    uint64_t file_size,
    uint64_t footer_size) {
  return std::make_shared<parquet::ParquetFormatReader>(fs, resolved_path, properties, needed_columns, key_retriever,
                                                        file_size, footer_size);
}

arrow::Result<std::unique_ptr<FormatWriter>> ParquetFormat::make_writer(
    const std::shared_ptr<arrow::fs::FileSystem>& fs,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& file_path,
    const std::shared_ptr<const api::Properties>& properties) {
  return parquet::ParquetFileWriter::Make(fs, schema, file_path, properties);
}

}  // namespace milvus_storage
