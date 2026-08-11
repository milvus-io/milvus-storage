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

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage::paimon {

class BlockingPaimonDataSplitReader;
class DataSplitStreamCursor;

class PaimonFormatReader final : public FormatReader {
  public:
  struct MetaTrait {
    struct Payload {
      std::string read_path;
      std::string data_format;
      // Direct-file reads delegate to the underlying Parquet/Vortex reader, so
      // the payload keeps that reader's parsed metadata (the alternative
      // matching data_format) to avoid reopening and reparsing the data file
      // each time a reader is created from this cached entry.
      std::variant<std::monostate,
                   parquet::ParquetFormatReader::MetaTrait::MetadataPtr,
                   vortex::VortexFormatReader::MetaTrait::MetadataPtr>
          direct_file_metadata;
      std::vector<RowGroupInfo> direct_physical_row_groups;
      std::shared_ptr<const std::vector<uint64_t>> sorted_deletions;
      // Shared projection-agnostic bridge handle for the data-split read
      // path. Creating it resolves the table schema and the pinned snapshot;
      // sharing it across every reader built from this cached metadata keeps
      // that cost per-descriptor instead of per-reader. Streams opened from
      // it are per-reader and per-projection.
      std::shared_ptr<BlockingPaimonDataSplitReader> split_reader_handle;
      uint64_t record_count = 0;
      uint64_t physical_row_count = 0;
    };

    using Metadata = FormatReaderMetadata<Payload>;
    using MetadataPtr = std::shared_ptr<const Metadata>;

    static std::string cache_key(const api::ColumnGroupFile& file);
    static arrow::Result<MetadataPtr> load_metadata(const api::ColumnGroupFile& file,
                                                    const api::Properties& properties,
                                                    const KeyRetriever& key_retriever);
    static arrow::Result<std::shared_ptr<PaimonFormatReader>> create_from_metadata(
        MetadataPtr metadata,
        const api::ColumnGroupFile& file,
        const std::shared_ptr<arrow::Schema>& read_schema,
        const std::vector<std::string>& needed_columns,
        const std::string& predicate);
  };

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;
  [[nodiscard]] arrow::Result<std::vector<uint64_t>> get_rg_column_memsz(int64_t row_group_index) const override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;
  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  PaimonFormatReader(MetaTrait::MetadataPtr metadata,
                     api::ColumnGroupFile file,
                     std::shared_ptr<arrow::Schema> read_schema,
                     std::vector<std::string> needed_columns,
                     std::shared_ptr<FormatReader> direct_file_reader,
                     std::shared_ptr<BlockingPaimonDataSplitReader> split_reader,
                     std::vector<std::string> split_columns,
                     std::shared_ptr<arrow::Schema> output_schema);
  [[nodiscard]] bool is_data_split() const;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> filter_direct_batch(
      const std::shared_ptr<arrow::RecordBatch>& batch, uint64_t physical_start) const;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> logical_to_physical(
      const std::vector<int64_t>& logical_offsets) const;
  [[nodiscard]] arrow::Result<std::unique_ptr<DataSplitStreamCursor>> make_data_split_cursor() const;

  MetaTrait::MetadataPtr metadata_;
  api::ColumnGroupFile file_;
  std::shared_ptr<arrow::Schema> read_schema_;
  std::vector<std::string> needed_columns_;
  std::shared_ptr<FormatReader> direct_file_reader_;
  // Shared with (and owned by) the cached metadata payload; streams opened
  // from it carry this reader's own projection in split_columns_.
  std::shared_ptr<BlockingPaimonDataSplitReader> split_reader_;
  std::vector<std::string> split_columns_;
  std::shared_ptr<arrow::Schema> output_schema_;
};

}  // namespace milvus_storage::paimon
