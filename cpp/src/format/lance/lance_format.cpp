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

#include "milvus-storage/format/lance/lance_format.h"

#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/lance/lance_common.h"
#include "milvus-storage/filesystem/fs.h"

#ifdef BUILD_GTEST
#include "milvus-storage/format/lance/lance_table_writer.h"
#endif

namespace milvus_storage {

arrow::Result<std::vector<api::ColumnGroupFile>> LanceFormat::explore(const std::string& explore_dir,
                                                                      const api::Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, explore_dir));
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, explore_dir.c_str()));

  ARROW_ASSIGN_OR_RAISE(auto explore_uri, StorageUri::Parse(explore_dir));

  ARROW_ASSIGN_OR_RAISE(auto lance_base_uri, lance::BuildLanceBaseUri(fs_config, explore_uri.key));
  ARROW_ASSIGN_OR_RAISE(auto snapshot_id, api::GetValue<int64_t>(properties, PROPERTY_READER_EXTTABLE_SNAPSHOT_ID));
  const auto requested_version = snapshot_id <= 0 ? uint64_t{0} : static_cast<uint64_t>(snapshot_id);
  // Deliberately bypass the process-wide Dataset cache: explore() establishes
  // the requested snapshot (latest for -1/0) and must not reuse an earlier
  // discovery. The actual version is persisted below for exact-version reads.
  ARROW_ASSIGN_OR_RAISE(auto dataset, lance::BlockingDataset::Open(
                                          lance_base_uri, fs, lance::ToReaderOptions(fs_config), requested_version));
  ARROW_ASSIGN_OR_RAISE(auto fragment_ids, dataset->GetAllFragmentIds());
  const auto dataset_version = dataset->Version();

  std::vector<api::ColumnGroupFile> files;
  for (auto frag_id : fragment_ids) {
    ARROW_ASSIGN_OR_RAISE(auto row_count, dataset->GetFragmentRowCount(frag_id));
    // Store Milvus-format URI (scheme://address/bucket/key) so the reader
    // can resolve the right extfs.<alias>.* by address+bucket. The reader
    // strips address back to standard form before handing to Lance.
    api::ColumnGroupFile file{
        lance::MakeLanceUri(lance::ToMilvusLanceUri(lance_base_uri, fs_config.address), frag_id),
        0,
        static_cast<int64_t>(row_count),
        {},
    };
    file.Set(lance::kDatasetVersionProperty, dataset_version);
    files.emplace_back(std::move(file));
  }

  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> LanceFormat::create_reader(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& /*key_retriever*/) {
  std::string base_path;
  uint64_t fragment_id;
  ARROW_ASSIGN_OR_RAISE(std::tie(base_path, fragment_id), lance::ParseLanceUri(file.path));
  ARROW_ASSIGN_OR_RAISE(auto fs, FilesystemCache::getInstance().get(properties, base_path));

  uint64_t dataset_version = 0;
  const auto version_it = file.properties.find(lance::kDatasetVersionProperty);
  if (version_it != file.properties.end()) {
    const auto [valid, version] = api::convert::convertFunc<uint64_t>(version_it->second);
    if (!valid) {
      return arrow::Status::Invalid("Invalid Lance dataset version for file ", file.path, ": ", version_it->second);
    }
    dataset_version = version;
  }

  auto reader = std::make_shared<lance::LanceTableReader>(fs, base_path, fragment_id, read_schema, properties,
                                                          needed_columns, dataset_version);
  ARROW_RETURN_NOT_OK(reader->open());
  return reader;
}

arrow::Result<std::unique_ptr<FormatWriter>> LanceFormat::create_writer(
    const std::shared_ptr<arrow::fs::FileSystem>& /*fs*/,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& /*file_path*/,
    const std::string& base_path,
    const api::Properties& properties) {
#ifdef BUILD_GTEST
  return std::make_unique<lance::LanceTableWriter>(base_path, schema, properties);
#else
  (void)schema;
  (void)base_path;
  (void)properties;
  return arrow::Status::NotImplemented("Lance writer is only available in test builds");
#endif
}

}  // namespace milvus_storage
