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

#pragma once
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"

namespace milvus_storage {

class PreparedParquetFile;

// Make opens independently; PreparedParquetFile provides explicit footer reuse.
class FileRowGroupReader {
  public:
  static arrow::Result<std::shared_ptr<FileRowGroupReader>> Make(
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const std::string& path,
      const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
      ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties(),
      const ::parquet::ArrowReaderProperties& arrow_reader_props = ::parquet::default_arrow_reader_properties());

  static arrow::Result<std::shared_ptr<FileRowGroupReader>> Make(
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const std::string& path,
      const std::shared_ptr<arrow::Schema> schema,
      const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
      ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties(),
      const ::parquet::ArrowReaderProperties& arrow_reader_props = ::parquet::default_arrow_reader_properties());

  arrow::Status SetRowGroupOffsetAndCount(int row_group_offset, int row_group_num);

  std::shared_ptr<arrow::Schema> schema() const;
  /**
   * @brief Returns packed file metadata.
   * Prepared readers share the Parquet footer; callers must treat it as read-only.
   */
  std::shared_ptr<PackedFileMetadata> file_metadata();

  /**
   * @brief Reads the next row group from the file.
   *
   * @param out A shared pointer to the output table.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status ReadNextRowGroup(std::shared_ptr<arrow::Table>* out);

  /**
   * @brief Closes the reader and releases resources.
   *
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status Close();

  protected:
  /**
   * @brief FileRowGroupReader reads specified row groups with memory constraints. The schema is the same as the file
   * schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param buffer_size Memory limit for reading row groups.
   * @param reader_props The reader properties.
   */
  FileRowGroupReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     const std::string& path,
                     const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
                     ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties());

  /**
   * @brief FileRowGroupReader reads specified row groups with memory constraints and schema.
   *
   * @param fs The Arrow filesystem interface.
   * @param path Path to the Parquet file.
   * @param schema The schema of data to read. If the field is not in the file, it will be filled with nulls.
   * @param buffer_size Memory limit for reading row groups.
   * @param reader_props The reader properties.
   */
  FileRowGroupReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                     const std::string& path,
                     const std::shared_ptr<arrow::Schema> schema,
                     const int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
                     ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties());

  private:
  friend class PreparedParquetFile;

  arrow::Status init(
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const std::string& path,
      const int64_t buffer_size,
      const std::shared_ptr<arrow::Schema> schema = nullptr,
      ::parquet::ReaderProperties reader_props = ::parquet::default_reader_properties(),
      const ::parquet::ArrowReaderProperties& arrow_reader_props = ::parquet::default_arrow_reader_properties(),
      std::shared_ptr<::parquet::FileMetaData> metadata = nullptr);

  /**
   * @brief Slices a row group from the table and updates the buffer state.
   *
   * @param table The input table to slice from.
   * @param buffer_size The current buffer size.
   * @param current_rg The current row group index.
   * @param out The output sliced table.
   * @return Arrow Status indicating success or failure.
   */
  arrow::Status SliceRowGroupFromTable(std::shared_ptr<arrow::Table>* out);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string path_;
  std::vector<int> needed_columns_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<::parquet::arrow::FileReader> file_reader_;
  FieldIDList field_id_list_;
  int rg_start_ = -1;
  int rg_end_ = -1;
  int current_rg_ = -1;

  int64_t buffer_size_limit_ = 0;
  int64_t buffer_size_ = 0;
  std::shared_ptr<PackedFileMetadata> file_metadata_;
  std::shared_ptr<arrow::Table> buffer_table_ = nullptr;
};

/**
 * Caller-owned context for an immutable packed Parquet file. Open once and
 * distribute to batch tasks by shared_ptr. Each MakeReader has its own file
 * handle, row-group range, cursor and data buffers. There is no global cache.
 *
 * The filesystem and original path are bound at Open; a rewritten file needs a
 * new context. Plaintext footers are shared read-only. Encrypted files are opened
 * fresh for every reader. Key retrievers/verifiers retained by Arrow DeepClone
 * must themselves support concurrent calls when readers are created in parallel.
 */
class PreparedParquetFile {
  public:
  static arrow::Result<std::shared_ptr<PreparedParquetFile>> Open(
      std::shared_ptr<arrow::fs::FileSystem> fs,
      const std::string& path,
      const ::parquet::ReaderProperties& reader_props = ::parquet::default_reader_properties());

  arrow::Result<std::shared_ptr<FileRowGroupReader>> MakeReader(
      const std::shared_ptr<arrow::Schema>& schema = nullptr,
      int64_t buffer_size = DEFAULT_READ_BUFFER_SIZE,
      const ::parquet::ArrowReaderProperties& arrow_reader_props = ::parquet::default_arrow_reader_properties()) const;

  const RowGroupMetadataVector& row_group_metadata() const { return row_group_metadata_; }
  const std::map<FieldID, ColumnOffset>& field_id_mapping() const { return field_id_mapping_; }
  const GroupFieldIDList& group_field_id_list() const { return group_field_id_list_; }
  const std::string& storage_version() const { return storage_version_; }

  private:
  PreparedParquetFile(std::shared_ptr<arrow::fs::FileSystem> fs,
                      std::string path,
                      const ::parquet::ReaderProperties& reader_props);

  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string path_;
  ::parquet::ReaderProperties reader_props_;
  // Null for encrypted files; never expose mutable shared metadata here.
  std::shared_ptr<::parquet::FileMetaData> plaintext_metadata_;
  RowGroupMetadataVector row_group_metadata_;
  std::map<FieldID, ColumnOffset> field_id_mapping_;
  GroupFieldIDList group_field_id_list_;
  std::string storage_version_;
};

}  // namespace milvus_storage
