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

#include <memory>
#include <string>
#include <iostream>
#include <regex>
#include <sstream>
#include <filesystem>

#include <fmt/core.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "milvus-storage/common/layout.h"
#include "milvus-storage/common/path_util.h"  // for kSep
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/properties.h"

namespace milvus_storage::api {

static inline std::filesystem::path generate_parent_path(const std::string& base_path) {
  std::filesystem::path path(base_path);
  return (path / kDataPath).lexically_normal();
}

static inline std::string generate_file_path(const std::string& base_path,
                                             const size_t& column_group_id,
                                             const std::string& format) {
  static boost::uuids::random_generator random_gen;
  boost::uuids::uuid random_uuid = random_gen();
  const std::string uuid_str = boost::uuids::to_string(random_uuid);
  // named as {group_id}_{uuid}.{format}
  const std::string file_name = fmt::format("{}_{}.{}", column_group_id, uuid_str, format);

  return (generate_parent_path(base_path) / file_name).lexically_normal().string();
}

// TODO(jiaqizho): implements file rolling in this class
class ColumnGroupWriterImpl final : public ColumnGroupWriter {
  public:
  ColumnGroupWriterImpl(const std::string& base_path,
                        const size_t& column_group_id,
                        const std::shared_ptr<ColumnGroup>& column_group,
                        const std::shared_ptr<arrow::Schema>& schema,
                        const Properties& properties)
      : base_path_(base_path),
        column_group_id_(column_group_id),
        column_group_(column_group),
        schema_(schema),
        properties_(properties) {}

  [[nodiscard]] arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override {
    return format_writer_->Write(record);
  }

  [[nodiscard]] arrow::Status Flush() override { return format_writer_->Flush(); }

  [[nodiscard]] arrow::Result<std::vector<ColumnGroupFile>> Close() override {
    ARROW_ASSIGN_OR_RAISE(auto column_group_file, format_writer_->Close());
    return std::vector<ColumnGroupFile>{std::move(column_group_file)};
  }

  [[nodiscard]] arrow::Status Open() override {
    assert(!format_writer_);
    ARROW_ASSIGN_OR_RAISE(auto writer,
                          create_format_writer(base_path_, column_group_id_, column_group_, schema_, properties_));
    format_writer_ = std::move(writer);
    return arrow::Status::OK();
  }

  private:
  static arrow::Result<std::unique_ptr<FormatWriter>> create_format_writer(
      const std::string& base_path,
      const size_t& column_group_id,
      const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
      const std::shared_ptr<arrow::Schema>& schema,
      const milvus_storage::api::Properties& properties) {
    std::unique_ptr<FormatWriter> writer;
    assert(column_group && schema);
    auto format = column_group->format;

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
    ARROW_ASSIGN_OR_RAISE(auto file_system, milvus_storage::FilesystemCache::getInstance().get(properties, base_path));

    // If current file system is local, create the parent directory if not exist
    // If current file system is remote, putobject will auto
    // create the parent directory if not exist
    if (IsLocalFileSystem(file_system)) {
      ARROW_RETURN_NOT_OK(file_system->CreateDir(generate_parent_path(base_path).string()));
    }

    if (format == LOON_FORMAT_PARQUET) {
      ARROW_ASSIGN_OR_RAISE(writer, parquet::ParquetFileWriter::Make(
                                        file_system, schema,
                                        std::move(generate_file_path(base_path, column_group_id, format)), properties));
    }
#ifdef BUILD_VORTEX_BRIDGE
    else if (format == LOON_FORMAT_VORTEX) {
      writer = std::make_unique<vortex::VortexFileWriter>(
          file_system, schema, std::move(generate_file_path(base_path, column_group_id, format)), properties);
    }
#endif  // BUILD_VORTEX_BRIDGE
    else {
      return arrow::Status::Invalid("Unsupported file format: " + format);
    }

    return writer;
  }

  private:
  std::unique_ptr<FormatWriter> format_writer_;

  std::string base_path_;
  size_t column_group_id_;
  std::shared_ptr<ColumnGroup> column_group_;
  std::shared_ptr<arrow::Schema> schema_;
  Properties properties_;
};

arrow::Result<std::unique_ptr<ColumnGroupWriter>> ColumnGroupWriter::create(
    const std::string& base_path,
    const size_t& column_group_id,
    const std::shared_ptr<ColumnGroup>& column_group,
    const std::shared_ptr<arrow::Schema>& schema,
    const Properties& properties) {
  std::unique_ptr<ColumnGroupWriterImpl> writer;
  assert(column_group && schema);
  writer = std::make_unique<ColumnGroupWriterImpl>(base_path, column_group_id, column_group, schema, properties);
  ARROW_RETURN_NOT_OK(writer->Open());
  return writer;
}

}  // namespace milvus_storage::api