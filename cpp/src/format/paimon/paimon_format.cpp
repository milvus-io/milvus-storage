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

#include "milvus-storage/format/paimon/paimon_format.h"

#include <nlohmann/json.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/paimon/paimon_common.h"
#include "milvus-storage/format/paimon/paimon_format_reader.h"
#include "paimon_bridge.h"

namespace milvus_storage {

arrow::Result<std::vector<api::ColumnGroupFile>> PaimonFormat::explore(const std::string& explore_dir,
                                                                       const api::Properties& properties) {
  ARROW_ASSIGN_OR_RAISE(auto fs_config, FilesystemCache::resolve_config(properties, explore_dir));
  ARROW_ASSIGN_OR_RAISE(auto scan_mode, api::GetValue<std::string>(properties, PROPERTY_PAIMON_SCAN_MODE));
  ARROW_ASSIGN_OR_RAISE(auto snapshot_id, api::GetValue<int64_t>(properties, PROPERTY_PAIMON_SNAPSHOT_ID));
  auto table_location = paimon::ToStandardUri(explore_dir);

  ARROW_ASSIGN_OR_RAISE(auto storage_options, paimon::ToStorageOptions(fs_config));
  ARROW_ASSIGN_OR_RAISE(auto planned, paimon::PlanFiles(table_location, snapshot_id, scan_mode, storage_options));

  std::vector<api::ColumnGroupFile> files;
  files.reserve(planned.size());
  for (const auto& info : planned) {
    int64_t record_count = -1;
    try {
      auto metadata = nlohmann::json::parse(info.metadata_json);
      if (!metadata.is_object()) {
        return arrow::Status::Invalid("Paimon planning metadata must be a JSON object");
      }
      if (metadata.value("read_path", std::string{}) != "direct-file") {
        return arrow::Status::Invalid("Paimon planner returned a non-direct metadata descriptor");
      }
      record_count = metadata.value("record_count", int64_t{-1});
    } catch (const nlohmann::json::exception& error) {
      return arrow::Status::Invalid("Invalid Paimon planning metadata: ", error.what());
    }
    if (record_count < 0) {
      return arrow::Status::Invalid("Paimon planning metadata has an invalid record_count");
    }
    if (record_count == 0) {
      continue;
    }
    api::ColumnGroupFile file{paimon::ToMilvusUri(info.path, fs_config.address), 0, record_count, {}};
    file.Set(api::kPropertyMetadata, info.metadata_json);
    if (info.file_size > 0) {
      file.Set(api::kPropertyFileSize, info.file_size);
    }
    files.push_back(std::move(file));
  }
  return files;
}

arrow::Result<std::shared_ptr<FormatReader>> PaimonFormat::create_reader(
    const std::shared_ptr<arrow::Schema>& read_schema,
    const api::ColumnGroupFile& file,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  ARROW_ASSIGN_OR_RAISE(auto metadata,
                        paimon::PaimonFormatReader::MetaTrait::load_metadata(file, properties, key_retriever));
  ARROW_ASSIGN_OR_RAISE(auto reader, paimon::PaimonFormatReader::MetaTrait::create_from_metadata(
                                         metadata, file, read_schema, needed_columns, ""));
  ARROW_RETURN_NOT_OK(reader->open());
  return std::static_pointer_cast<FormatReader>(reader);
}

arrow::Result<std::unique_ptr<FormatWriter>> PaimonFormat::create_writer(const std::shared_ptr<arrow::fs::FileSystem>&,
                                                                         const std::shared_ptr<arrow::Schema>&,
                                                                         const std::string&,
                                                                         const std::string&,
                                                                         const api::Properties&) {
  return arrow::Status::NotImplemented("Paimon format does not support writing");
}

}  // namespace milvus_storage
