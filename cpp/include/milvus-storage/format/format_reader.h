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

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/record_batch.h>

#include "milvus-storage/properties.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/column_groups.h"

namespace milvus_storage {

using KeyRetriever = std::function<std::string(const std::string&)>;

struct RowGroupInfo {
  public:
  size_t start_offset;
  size_t end_offset;
  size_t memory_size;

  std::string ToString() const;
};

template <typename Payload>
struct FormatReaderMetadata {
  std::string cache_key;
  std::string path;
  std::shared_ptr<arrow::Schema> file_schema;
  std::vector<RowGroupInfo> row_group_infos;
  uint64_t cache_size = 0;
  Payload payload;
};

/**
 * FormatReader is a reader to read the format file.
 * It exists both blocking and streaming read interfaces.
 * Blocking interface:
 *   - get_chunk
 *   - get_chunks
 *   - take
 * Streaming interface:
 *   - read_with_range
 *
 */
class FormatReader {
  public:
  template <typename ReaderT>
  using MetaTrait = typename ReaderT::MetaTrait;

  template <typename ReaderT>
  using MetadataPtr = typename MetaTrait<ReaderT>::MetadataPtr;

  virtual ~FormatReader() = default;

  // open the format reader, usage to initialize the reader
  // `open` is typically used to open the file's footer.
  [[nodiscard]] virtual arrow::Status open() = 0;

  // get the row group infos
  [[nodiscard]] virtual arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() = 0;

  // get the chunk
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) = 0;

  // get the chunks
  [[nodiscard]] virtual arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) = 0;

  // take the rows, row_indices MUST be uniqued and sorted
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) = 0;

  // create a streaming reader to read the rows in the range
  //
  // If current format reader support `get_row_group_infos`,
  // then the response from the streaming reader should no
  // longer be split from the middle of the row group. Otherwise,
  // there will be an additional memory copy.
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) = 0;

  // clone itself for multi-threading
  // if the reader is not thread-safe, then it should be cloned
  // if the reader is thread-safe, then return itself
  [[nodiscard]] virtual arrow::Result<std::shared_ptr<FormatReader>> clone_reader() = 0;

  // get the file schema of this reader (always derived from file metadata, not projected)
  [[nodiscard]] virtual std::shared_ptr<arrow::Schema> get_schema() const = 0;

  // set a predicate string for filtering (default no-op for formats that don't support it)
  [[nodiscard]] virtual arrow::Status set_predicate(const std::string& /*predicate*/) { return arrow::Status::OK(); }

  // Load reusable file metadata without applying read-time state such as
  // projection or predicate. The returned metadata is safe to share through
  // MetadataCache and later reuse to create independent readers.
  template <typename ReaderT>
  static arrow::Result<MetadataPtr<ReaderT>> load_metadata(const api::ColumnGroupFile& file,
                                                           const api::Properties& properties,
                                                           const KeyRetriever& key_retriever);

  // Create a new stateful reader from cached metadata. The file carries
  // manifest-owned values such as file_size and footer_size; read_schema,
  // needed_columns, and predicate are applied here so callers can create
  // independent readers with different projections or filters from the same
  // cached metadata.
  template <typename ReaderT>
  static arrow::Result<std::shared_ptr<ReaderT>> create_from_metadata(MetadataPtr<ReaderT> metadata,
                                                                      const api::ColumnGroupFile& file,
                                                                      const std::shared_ptr<arrow::Schema>& read_schema,
                                                                      const std::vector<std::string>& needed_columns,
                                                                      const std::string& predicate);

  // create format reader
  static arrow::Result<std::shared_ptr<FormatReader>> create(
      const std::shared_ptr<arrow::Schema>& read_schema,
      const std::string& format,
      const api::ColumnGroupFile& file,
      const api::Properties& properties,
      const std::vector<std::string>& needed_columns,
      const std::function<std::string(const std::string&)>& key_retriever);

};  // class FormatReader

template <typename ReaderT>
concept FormatReaderWithMetadata =
    std::derived_from<ReaderT, FormatReader> && requires(const api::ColumnGroupFile& file,
                                                         const api::Properties& properties,
                                                         const KeyRetriever& key_retriever,
                                                         typename ReaderT::MetaTrait::MetadataPtr metadata,
                                                         const api::ColumnGroupFile& metadata_file,
                                                         const std::shared_ptr<arrow::Schema>& read_schema,
                                                         const std::vector<std::string>& needed_columns,
                                                         const std::string& predicate) {
      typename ReaderT::MetaTrait::Payload;
      typename ReaderT::MetaTrait::Metadata;
      typename ReaderT::MetaTrait::MetadataPtr;

      requires std::same_as<typename ReaderT::MetaTrait::Metadata,
                            FormatReaderMetadata<typename ReaderT::MetaTrait::Payload>>;
      requires std::same_as<typename ReaderT::MetaTrait::MetadataPtr,
                            std::shared_ptr<const typename ReaderT::MetaTrait::Metadata>>;

      { ReaderT::MetaTrait::cache_key(file) } -> std::convertible_to<std::string>;
      {
        ReaderT::MetaTrait::load_metadata(file, properties, key_retriever)
      } -> std::same_as<arrow::Result<typename ReaderT::MetaTrait::MetadataPtr>>;
      {
        ReaderT::MetaTrait::create_from_metadata(metadata, metadata_file, read_schema, needed_columns, predicate)
      } -> std::same_as<arrow::Result<std::shared_ptr<ReaderT>>>;
      { metadata->row_group_infos } -> std::same_as<const std::vector<RowGroupInfo>&>;
      { metadata->file_schema } -> std::same_as<const std::shared_ptr<arrow::Schema>&>;
    };

template <typename ReaderT>
arrow::Result<FormatReader::MetadataPtr<ReaderT>> FormatReader::load_metadata(const api::ColumnGroupFile& file,
                                                                              const api::Properties& properties,
                                                                              const KeyRetriever& key_retriever) {
  static_assert(FormatReaderWithMetadata<ReaderT>,
                "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                "cache_key, load_metadata, and create_from_metadata.");
  return ReaderT::MetaTrait::load_metadata(file, properties, key_retriever);
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<ReaderT>> FormatReader::create_from_metadata(
    MetadataPtr<ReaderT> metadata,
    const api::ColumnGroupFile& file,
    const std::shared_ptr<arrow::Schema>& read_schema,
    const std::vector<std::string>& needed_columns,
    const std::string& predicate) {
  static_assert(FormatReaderWithMetadata<ReaderT>,
                "ReaderT must derive from FormatReader and define MetaTrait with Payload, Metadata, MetadataPtr, "
                "cache_key, load_metadata, and create_from_metadata.");
  return ReaderT::MetaTrait::create_from_metadata(metadata, file, read_schema, needed_columns, predicate);
}

}  // namespace milvus_storage
