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

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/parquet_chunk_reader.h"
#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/format/vortex/vortex_chunk_reader.h"
#include "milvus-storage/properties.h"

namespace internal::api {

using namespace milvus_storage::parquet;
static inline arrow::Result<milvus_storage::ArrowFileSystemPtr> create_arrow_file_system(
    const milvus_storage::ArrowFileSystemConfig& fs_config) {
  auto& fs_cache = milvus_storage::LRUCache<milvus_storage::ArrowFileSystemConfig,
                                            milvus_storage::ArrowFileSystemPtr>::getInstance();
  return fs_cache.get(fs_config, milvus_storage::CreateArrowFileSystem);
}

#ifdef BUILD_VORTEX_BRIDGE
using namespace milvus_storage::vortex;
#endif  // BUILD_VORTEX_BRIDGE

// ==================== ColumnGroupReaderFactory Implementation ====================

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever) {
  std::unique_ptr<ColumnGroupReader> reader = nullptr;
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  // Generate the output schema with only the needed columns
  std::vector<std::string> filtered_columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
      auto field = schema->GetFieldByName(col_name);
      assert(field);
      fields.emplace_back(field);
    }
  }

  std::shared_ptr<arrow::Schema> out_schema = std::make_shared<arrow::Schema>(fields);

  milvus_storage::ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
  ARROW_ASSIGN_OR_RAISE(auto file_system, create_arrow_file_system(fs_config));
  if (column_group->format == LOON_FORMAT_PARQUET) {
    ::parquet::ReaderProperties reader_properties = ::parquet::default_reader_properties();
    if (key_retriever) {
      reader_properties.file_decryption_properties(::parquet::FileDecryptionProperties::Builder()
                                                       .key_retriever(std::make_shared<KeyRetriever>(key_retriever))
                                                       ->plaintext_files_allowed()
                                                       ->build());
    }

    reader = std::make_unique<ParquetChunkReader>(file_system, column_group, reader_properties, filtered_columns);
  }
#ifdef BUILD_VORTEX_BRIDGE
  else if (column_group->format == LOON_FORMAT_VORTEX) {
    reader = std::make_unique<VortexChunkReader>(file_system, out_schema, column_group, properties, filtered_columns);
  }
#endif  // BUILD_VORTEX_BRIDGE
  else {
    return arrow::Status::Invalid("Unsupported file format: " + column_group->format);
  }

  ARROW_RETURN_NOT_OK(reader->open());
  return std::move(reader);
}

// ==================== ChunkWriterFactory Implementation ====================

arrow::Result<std::unique_ptr<ColumnGroupWriter>> ColumnGroupWriter::create(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::Schema> schema,
    const milvus_storage::api::Properties& properties) {
  std::unique_ptr<ColumnGroupWriter> writer;
  assert(column_group && schema);

  // Create schema with only the columns for this column group
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& column_name : column_group->columns) {
    auto field = schema->GetFieldByName(column_name);
    if (!field) {
      return arrow::Status::Invalid("Column '" + column_name + "' not found in schema");
    }
    fields.emplace_back(field);
  }
  auto column_group_schema = arrow::schema(fields);

  // create the file system by cache
  milvus_storage::ArrowFileSystemConfig fs_config;
  ARROW_RETURN_NOT_OK(milvus_storage::ArrowFileSystemConfig::create_file_system_config(properties, fs_config));
  ARROW_ASSIGN_OR_RAISE(auto file_system, create_arrow_file_system(fs_config));
  if (column_group->format == LOON_FORMAT_PARQUET) {
    writer = std::make_unique<ParquetFileWriter>(column_group, file_system, schema, properties);
  }
#ifdef BUILD_VORTEX_BRIDGE
  else if (column_group->format == LOON_FORMAT_VORTEX) {
    writer = std::make_unique<VortexFileWriter>(column_group, file_system, schema, properties);
  }
#endif  // BUILD_VORTEX_BRIDGE
  else {
    return arrow::Status::Invalid("Unsupported file format: " + column_group->format);
  }
  return writer;
}

}  // namespace internal::api