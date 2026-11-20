// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

#include "deltalake_test_util.h"

#include <vector>

#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include "deltalake/deltalake_bridge.h"
#include "test_env.h"

namespace milvus_storage::deltalake::test {

arrow::Result<std::shared_ptr<arrow::Schema>> WriteTable(const ArrowFileSystemPtr& fs,
                                                         const std::string& path,
                                                         int64_t rows_per_file,
                                                         int file_count,
                                                         std::span<const int64_t> deleted_rows) {
  using Json = nlohmann::json;
  constexpr int64_t rows_per_group = 8192;
  if (IsLocalFileSystem(fs)) {
    ARROW_RETURN_NOT_OK(fs->CreateDir(path + "/_delta_log", true));
  }
  ARROW_ASSIGN_OR_RAISE(auto schema, CreateTestSchema({true, true, true, false}));

  Json protocol{{"minReaderVersion", 1}, {"minWriterVersion", 2}};
  Json configuration = Json::object();
  Json deletion_vector;
  if (!deleted_rows.empty()) {
    const std::vector<uint64_t> positions(deleted_rows.begin(), deleted_rows.end());
    const auto dv = ffi::delta_test_deletion_vector({positions.data(), positions.size()});
    ARROW_ASSIGN_OR_RAISE(auto sink,
                          fs->OpenOutputStream(path + "/deletion_vector_00000000-0000-0000-0000-000000000000.bin"));
    ARROW_RETURN_NOT_OK(sink->Write(dv.bytes.data(), dv.bytes.size()));
    ARROW_RETURN_NOT_OK(sink->Close());
    protocol = Json{{"minReaderVersion", 3},
                    {"minWriterVersion", 7},
                    {"readerFeatures", {"deletionVectors"}},
                    {"writerFeatures", {"deletionVectors"}}};
    configuration["delta.enableDeletionVectors"] = "true";
    deletion_vector = Json{{"storageType", "u"},
                           {"pathOrInlineDv", std::string(20, '0')},  // Z85 for the all-zero UUID.
                           {"offset", dv.offset},
                           {"sizeInBytes", dv.size},
                           {"cardinality", dv.cardinality}};
  }

  Json fields = Json::array();
  const std::vector<std::string> types{"long", "string", "double"};
  for (int index = 0; index < schema->num_fields(); ++index) {
    const auto& field = schema->field(index);
    fields.push_back(Json{{"name", field->name()},
                          {"type", types[index]},
                          {"nullable", field->nullable()},
                          {"metadata", Json::object()}});
  }
  const Json delta_schema{{"type", "struct"}, {"fields", std::move(fields)}};
  const Json metadata{{"id", "bf692c52-1384-41b8-a8d0-99413a3400bc"},
                      {"format", {{"provider", "parquet"}, {"options", Json::object()}}},
                      {"schemaString", delta_schema.dump()},
                      {"partitionColumns", Json::array()},
                      {"configuration", std::move(configuration)}};
  auto commit = Json{{"protocol", std::move(protocol)}}.dump() + '\n' + Json{{"metaData", metadata}}.dump() + '\n';
  for (int file = 0; file < file_count; ++file) {
    const auto name = "part-" + std::to_string(file) + ".parquet";
    ARROW_ASSIGN_OR_RAISE(auto batch, CreateTestData(schema, file * rows_per_file, false, rows_per_file, 4, 50,
                                                     {true, true, true, false}));
    ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches({batch}));
    ARROW_ASSIGN_OR_RAISE(auto sink, fs->OpenOutputStream(path + "/" + name));
    ARROW_RETURN_NOT_OK(::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, rows_per_group));
    ARROW_ASSIGN_OR_RAISE(auto size, sink->Tell());
    ARROW_RETURN_NOT_OK(sink->Close());

    Json stats{{"numRecords", rows_per_file}};
    Json add{{"path", name},
             {"size", size},
             {"partitionValues", Json::object()},
             {"modificationTime", 0},
             {"dataChange", true}};
    if (file == 0 && !deleted_rows.empty()) {
      add["deletionVector"] = deletion_vector;
      stats["tightBounds"] = false;
    }
    add["stats"] = stats.dump();
    commit += Json{{"add", std::move(add)}}.dump() + '\n';
  }

  ARROW_ASSIGN_OR_RAISE(auto log, fs->OpenOutputStream(path + "/_delta_log/00000000000000000000.json"));
  ARROW_RETURN_NOT_OK(log->Write(commit));
  ARROW_RETURN_NOT_OK(log->Close());
  return schema;
}

arrow::Result<std::string> ExploreUri(const ArrowFileSystemConfig& config, const std::string& path) {
  if (config.storage_type == "local") {
    return path;
  }
  StorageUri uri;
  uri.scheme = "s3";
  if (config.cloud_provider == kCloudProviderAzure) {
    uri.scheme = "az";
  } else if (config.cloud_provider == kCloudProviderGCP) {
    uri.scheme = "gs";
  } else if (config.cloud_provider == kCloudProviderAliyun) {
    uri.scheme = "oss";
  }
  uri.address = config.address;
  uri.bucket_name = config.bucket_name;
  uri.key = path;
  return StorageUri::Make(uri);
}

}  // namespace milvus_storage::deltalake::test
