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
#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "lance_bridge.h"  // from cpp/src/format/lance/lance-bridge/src/include

namespace milvus_storage::lance {

class LanceTableReader final : public FormatReader, public std::enable_shared_from_this<LanceTableReader> {
  public:
  LanceTableReader(const std::string& uri,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const milvus_storage::api::Properties& properties,
                   const std::vector<std::string>& needed_columns = {},
                   uint64_t dataset_version = 0);

  struct MetaTrait {
    // One outer Metadata entry represents a Lance Dataset and is keyed by its
    // base URI. Files for that URI in one top-level reader must reference the
    // same Dataset version; create_from_metadata() enforces this invariant.
    // Fragment-specific immutable metadata is cached inside Payload.
    struct FragmentMetadata;
    class FragmentMetadataCache;

    struct Payload {
      std::string base_uri;
      std::shared_ptr<BlockingDataset> dataset;
      // BlockingFragmentReader is intentionally not cached here because it is
      // projection-specific and stateful.
      std::shared_ptr<FragmentMetadataCache> fragment_metadata_cache;
      milvus_storage::api::Properties properties;
    };

    using Metadata = FormatReaderMetadata<Payload>;
    using MetadataPtr = std::shared_ptr<const Metadata>;

    static std::string cache_key(const milvus_storage::api::ColumnGroupFile& file);

    static arrow::Result<MetadataPtr> load_metadata(const milvus_storage::api::ColumnGroupFile& file,
                                                    const milvus_storage::api::Properties& properties,
                                                    const KeyRetriever& key_retriever);

    static arrow::Result<std::shared_ptr<LanceTableReader>> create_from_metadata(
        MetadataPtr metadata,
        const milvus_storage::api::ColumnGroupFile& file,
        const std::shared_ptr<arrow::Schema>& read_schema,
        const std::vector<std::string>& needed_columns,
        const std::string& predicate);
  };

  [[nodiscard]] arrow::Status open() override;

  // get the row group infos
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;

  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_rg_column_memsz(int64_t row_group_index) const override;

  // get the chunk
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;

  // get the chunks
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;

  // take
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;

  // read with range
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  LanceTableReader(MetaTrait::MetadataPtr metadata,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const std::vector<std::string>& needed_columns = {});

  std::shared_ptr<BlockingDataset> dataset_;
  std::string uri_;
  uint64_t fragment_id_;
  uint64_t dataset_version_ = 0;
  std::shared_ptr<arrow::Schema> read_schema_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;

  std::shared_ptr<arrow::Schema> file_schema_;  // always derived from fragment metadata in open()

  uint64_t logical_chunk_rows_;
  uint64_t num_deletions_ = 0;  // physical_rows - logical_rows
  std::shared_ptr<const std::vector<uint64_t>> column_memory_weights_;
  std::vector<RowGroupInfo> row_group_infos_;
  std::unique_ptr<BlockingFragmentReader> fragment_reader_;
};

}  // namespace milvus_storage::lance
