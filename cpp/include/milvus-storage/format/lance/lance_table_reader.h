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
#include <optional>
#include <string>
#include <vector>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "lance_bridge.h"  // from cpp/src/format/lance/lance-bridge/src/include

namespace milvus_storage::lance {

class LanceTableReader final : public FormatReader, public std::enable_shared_from_this<LanceTableReader> {
  public:
  LanceTableReader(const std::shared_ptr<BlockingDataset>& dataset,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const milvus_storage::api::Properties& properties,
                   const std::vector<std::string>& needed_columns = {});

  LanceTableReader(const std::string& uri,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const milvus_storage::api::Properties& properties,
                   const std::vector<std::string>& needed_columns = {});

  struct MetaTrait {
    struct Payload {
      std::string base_uri;
      uint64_t fragment_id = 0;
      std::shared_ptr<BlockingDataset> dataset;
      uint64_t logical_row_count = 0;
      uint64_t physical_row_count = 0;
      uint64_t num_deletions = 0;
      uint64_t logical_chunk_rows = 0;
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

  // Per-column (projected) size weights of a single chunk, from footer/page metadata only.
  // Lance's estimator reports per-fragment decoded (in-memory) sizes; the whole-fragment
  // per-column estimate is returned as raw weights (no per-chunk pro-rata here) and
  // normalized by ColumnGroupReader.
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_column_sizes(int row_group_index) override;

  private:
  std::shared_ptr<BlockingDataset> dataset_;
  std::string uri_;
  uint64_t fragment_id_;
  std::shared_ptr<arrow::Schema> read_schema_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;

  std::shared_ptr<arrow::Schema> file_schema_;  // always derived from fragment metadata in open()

  uint64_t logical_chunk_rows_;
  uint64_t num_deletions_ = 0;  // physical_rows - logical_rows
  std::vector<RowGroupInfo> row_group_infos_;
  std::unique_ptr<BlockingFragmentReader> fragment_reader_;

  // Cached per-top-level-column decoded (in-memory) estimate for this fragment, in file-schema
  // field order. The Lance estimator runs a metadata scan, so we compute it at most once and
  // reuse it across get_column_sizes() calls. An empty vector means the estimator could not
  // run (report no weights); a nullopt means "not yet computed".
  std::optional<std::vector<uint64_t>> fragment_column_memory_;
};

}  // namespace milvus_storage::lance
