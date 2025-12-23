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

#include "milvus-storage/format/column_group_writer.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/properties.h"

namespace milvus_storage::api {

arrow::Result<std::unique_ptr<ColumnGroupWriter>> ColumnGroupWriter::create(
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::shared_ptr<arrow::Schema>& schema,
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
  // Use first file path if available, otherwise use empty string for default filesystem
  if (UNLIKELY(column_group->files.empty())) {
    return arrow::Status::Invalid("Logical fault, column group files is empty");
  }

  std::string path = column_group->files[0].path;
  ARROW_ASSIGN_OR_RAISE(auto file_system, milvus_storage::FilesystemCache::getInstance().get(properties, path));

  // If current file system is local, create the parent directory if not exist
  // If current file system is remote, putobject will auto
  // create the parent directory if not exist
  if (IsLocalFileSystem(file_system)) {
    boost::filesystem::path dir_path(path);
    auto parent_dir_path = dir_path.parent_path();
    ARROW_RETURN_NOT_OK(file_system->CreateDir(parent_dir_path.string()));
  }

  if (column_group->format == LOON_FORMAT_PARQUET) {
    ARROW_ASSIGN_OR_RAISE(
        writer, milvus_storage::parquet::ParquetFileWriter::Make(column_group, file_system, schema, properties));
  }
#ifdef BUILD_VORTEX_BRIDGE
  else if (column_group->format == LOON_FORMAT_VORTEX) {
    writer = std::make_unique<vortex::VortexFileWriter>(column_group, file_system, schema, properties);
  }
#endif  // BUILD_VORTEX_BRIDGE
  else {
    return arrow::Status::Invalid("Unsupported file format: " + column_group->format);
  }
  return writer;
}

}  // namespace milvus_storage::api