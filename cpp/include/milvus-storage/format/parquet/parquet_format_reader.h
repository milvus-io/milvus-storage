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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/schema.h>
#include <parquet/type_fwd.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::parquet {

class ParquetFormatReader final : public FormatReader, public std::enable_shared_from_this<ParquetFormatReader> {
  public:
  struct MetaTrait {
    struct Payload {
      std::shared_ptr<arrow::fs::FileSystem> fs;
      std::shared_ptr<::parquet::FileMetaData> parquet_metadata;
      api::Properties properties;
      milvus_storage::KeyRetriever key_retriever;
    };

    using Metadata = FormatReaderMetadata<Payload>;
    using MetadataPtr = std::shared_ptr<const Metadata>;

    static std::string cache_key(const api::ColumnGroupFile& file);
    static arrow::Result<MetadataPtr> load_metadata(const api::ColumnGroupFile& file,
                                                    const api::Properties& properties,
                                                    const milvus_storage::KeyRetriever& key_retriever);
    // Defer footer/schema loading and return immutable metadata suitable for the cache.
    static folly::SemiFuture<arrow::Result<MetadataPtr>> load_metadata_async(
        const api::ColumnGroupFile& file,
        const api::Properties& properties,
        const milvus_storage::KeyRetriever& key_retriever);
    static arrow::Result<std::shared_ptr<ParquetFormatReader>> create_from_metadata(
        MetadataPtr metadata,
        const api::ColumnGroupFile& file,
        const std::shared_ptr<arrow::Schema>& read_schema,
        const std::vector<std::string>& needed_columns,
        const std::string& predicate);

 private:
    // Implementation detail, not part of FormatReaderWithMetadata. This helper
    // lives in the nested MetaTrait only to access the reader's private state.
    static arrow::Result<MetadataPtr> create_metadata_from_reader(const std::shared_ptr<ParquetFormatReader>& reader,
                                                                  const api::ColumnGroupFile& file);
  };

  ParquetFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                      const std::string& path,
                      const milvus_storage::api::Properties& properties,
                      const std::vector<std::string>& needed_columns,
                      const std::function<std::string(const std::string&)>& key_retriever,
                      uint64_t file_size = 0,
                      uint64_t footer_size = 0);

  // open the file
  [[nodiscard]] arrow::Status open() override;
  // Defer open() until future consumption and retain this reader until completion.
  [[nodiscard]] folly::SemiFuture<arrow::Status> open_async() override;

  // get the row group infos
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;

  // get the chunk
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;

  // get the chunks
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;

  // get the chunk indices
  [[nodiscard]] arrow::Result<std::vector<int>> get_chunk_indices(const std::vector<int64_t>& row_indices);

  // take the rows
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;

  // read with range
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;

  // Decode overlapping row groups with Arrow's async generator, then trim to
  // the exact file-local half-open range. Arrow work inherits the consumer executor.
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>> read_with_range_async(
      uint64_t start_offset, uint64_t end_offset) override;

  // Decode each touched row group once, then remap the requested file-local rows.
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(
      const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunks_internal(
      const std::vector<int>& rg_indices_in_file);
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> create_row_group_infos(
      const std::shared_ptr<::parquet::FileMetaData>& metadata);
  [[nodiscard]] arrow::Status set_needed_columns(const std::vector<std::string>& needed_columns);

  ParquetFormatReader(const ParquetFormatReader& other, std::shared_ptr<::parquet::arrow::FileReader> file_reader);

  std::string path_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  uint64_t file_size_ = 0;    ///< Pre-known file size to skip S3 HEAD requests
  uint64_t footer_size_ = 0;  ///< Pre-known footer size for single-IO footer read

  // init after open()
  // Parquet ReadRowGroup expects leaf-column indices, not top-level Arrow field indices.
  std::vector<int> projected_leaf_column_indices_;
  std::vector<RowGroupInfo> row_group_infos_;
  std::shared_ptr<::parquet::arrow::FileReader> file_reader_;
};  // ParquetFormatReader

}  // namespace milvus_storage::parquet
