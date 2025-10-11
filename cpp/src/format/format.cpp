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
#include "milvus-storage/format/parquet/file_writer.h"
#include "milvus-storage/format/parquet/reader.h"
#include "milvus-storage/properties.h"

namespace internal::api {

// ==================== ColumnGroupReaderFactory Implementation ====================

std::unique_ptr<ColumnGroupReader> GroupReaderFactory::create(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties) {
  std::unique_ptr<ColumnGroupReader> reader;
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.push_back(col_name);
    }
  }

  if (column_group->format == "parquet") {
    reader = std::make_unique<milvus_storage::parquet::ParquetChunkReader>(
        fs, schema, column_group->paths, parquet::default_reader_properties(), filtered_columns);
  } else {
    throw std::runtime_error("Unsupported file format: " + column_group->format + ". Only PARQUET is supported.");
  }

  auto status = reader->open();
  if (!status.ok()) {
    throw std::runtime_error("Error opening column group reader: " + status.ToString());
  }
  return reader;
}

// ==================== ChunkWriterFactory Implementation ====================

std::unique_ptr<ColumnGroupWriter> GroupWriterFactory::create(
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::fs::FileSystem> fs,
    const milvus_storage::api::Properties& properties) {
  if (!column_group) {
    throw std::runtime_error("Column group cannot be null");
  }

  if (!schema) {
    throw std::runtime_error("Schema cannot be null");
  }

  // Create schema with only the columns for this column group
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& column_name : column_group->columns) {
    auto field = schema->GetFieldByName(column_name);
    if (!field) {
      throw std::runtime_error("Column '" + column_name + "' not found in schema");
    }
    fields.push_back(field);
  }
  auto column_group_schema = arrow::schema(fields);

  std::unique_ptr<internal::api::ColumnGroupWriter> writer;

  if (column_group->format == "parquet") {
    writer = std::make_unique<milvus_storage::parquet::ParquetFileWriter>(column_group, fs, schema, properties);
  } else {
    throw std::runtime_error("Unsupported file format: " + column_group->format + ". Only PARQUET is supported.");
  }

  return writer;
}

}  // namespace internal::api