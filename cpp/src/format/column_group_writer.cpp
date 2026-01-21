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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/layout.h"
#include "milvus-storage/common/path_util.h"  // for kSep
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/parquet/parquet_writer.h"
#include "milvus-storage/format/parquet/key_retriever.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/format/lance/lance_table_writer.h"
#include <fmt/format.h>
#include "milvus-storage/properties.h"

namespace milvus_storage::api {

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
        properties_(properties),
        format_writer_(nullptr),
        max_bytes_limit_(0),
        written_bytes_(0) {}

  [[nodiscard]] arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override {
    written_bytes_ += GetRecordBatchMemorySize(record);
    return format_writer_->Write(record);
  }

  [[nodiscard]] arrow::Status Flush() override {
    ARROW_RETURN_NOT_OK(format_writer_->Flush());

    if (written_bytes_ == 0 || written_bytes_ < max_bytes_limit_) {
      return arrow::Status::OK();
    }

    // rolling to next file
    ARROW_ASSIGN_OR_RAISE(auto column_group_file, format_writer_->Close());
    written_files_.emplace_back(std::move(column_group_file));
    written_bytes_ = 0;

    ARROW_ASSIGN_OR_RAISE(format_writer_,
                          create_format_writer(base_path_, column_group_id_, column_group_, schema_, properties_));

    return arrow::Status::OK();
  }

  [[nodiscard]] arrow::Result<std::vector<ColumnGroupFile>> Close() override {
    assert(format_writer_);
    ARROW_ASSIGN_OR_RAISE(auto column_group_file, format_writer_->Close());
    // If current column_group_file without any data, do not add it to written_files_
    if (written_bytes_ != 0) {
      written_files_.emplace_back(std::move(column_group_file));
    }
    return written_files_;
  }

  [[nodiscard]] arrow::Status Open() override {
    assert(!format_writer_);
    ARROW_ASSIGN_OR_RAISE(format_writer_,
                          create_format_writer(base_path_, column_group_id_, column_group_, schema_, properties_));
    ARROW_ASSIGN_OR_RAISE(max_bytes_limit_, api::GetValue<uint64_t>(properties_, PROPERTY_WRITER_FILE_ROLLING_SIZE));
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
        return arrow::Status::Invalid(fmt::format("Column '{}' not found in schema. [base_path={}, column_group_id={}]",
                                                  column_name, base_path, column_group_id));
      }
      fields.emplace_back(field);
    }
    auto column_group_schema = arrow::schema(fields);
    ARROW_ASSIGN_OR_RAISE(auto file_system, milvus_storage::FilesystemCache::getInstance().get(properties, base_path));

    // If current file system is local, create the parent directory if not exist
    // If current file system is remote, putobject will auto
    // create the parent directory if not exist
    if (IsLocalFileSystem(file_system)) {
      ARROW_RETURN_NOT_OK(file_system->CreateDir(get_data_path(base_path)));
    }

    if (format == LOON_FORMAT_PARQUET) {
      ARROW_ASSIGN_OR_RAISE(writer, parquet::ParquetFileWriter::Make(
                                        file_system, schema,
                                        std::move(get_data_filepath(base_path, column_group_id, format)), properties));
    }
#ifdef BUILD_VORTEX_BRIDGE
    else if (format == LOON_FORMAT_VORTEX) {
      writer = std::make_unique<vortex::VortexFileWriter>(
          file_system, schema, std::move(get_data_filepath(base_path, column_group_id, format)), properties);
    }
#endif  // BUILD_VORTEX_BRIDGE
#ifdef BUILD_LANCE_BRIDGE
#ifdef BUILD_GTEST
    else if (format == LOON_FORMAT_LANCE_TABLE) {
      writer = std::make_unique<lance::LanceTableWriter>(base_path, schema, properties);
    }
#endif  // BUILD_GTEST
#endif  // BUILD_LANCE_BRIDGE
    else {
      return arrow::Status::Invalid(fmt::format("Unknown file format: '{}'. [base_path={}, column_group_id={}]", format,
                                                base_path, column_group_id));
    }

    return writer;
  }

  private:
  std::string base_path_;
  size_t column_group_id_;
  std::shared_ptr<ColumnGroup> column_group_;
  std::shared_ptr<arrow::Schema> schema_;
  Properties properties_;

  std::unique_ptr<FormatWriter> format_writer_;

  uint64_t max_bytes_limit_;
  uint64_t written_bytes_;

  std::vector<ColumnGroupFile> written_files_;
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