// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz
#pragma once

#include "milvus-storage/format/format_reader.h"

namespace milvus_storage::parquet {
class ParquetFormatReader;
}

namespace milvus_storage::deltalake {

class DeltaTableContext;
class DeltaFileCache;
struct DeltaFileMetadata;
class BlockingDeltaReadPlan;

class DeltaLakeFormatReader final : public FormatReader, public std::enable_shared_from_this<DeltaLakeFormatReader> {
  public:
  struct MetaTrait {
    struct Payload {
      std::shared_ptr<DeltaTableContext> table;
      std::shared_ptr<DeltaFileCache> files;
      std::shared_ptr<const api::Properties> properties;
      KeyRetriever key_retriever;
    };
    using Metadata = FormatReaderMetadata<Payload>;
    using MetadataPtr = std::shared_ptr<const Metadata>;
    static std::string cache_key(const api::ColumnGroupFile& file);
    static arrow::Result<MetadataPtr> load_metadata(const api::ColumnGroupFile& file,
                                                    const std::shared_ptr<const api::Properties>& properties,
                                                    const KeyRetriever& key_retriever);
    static arrow::Result<std::shared_ptr<DeltaLakeFormatReader>> create_from_metadata(
        MetadataPtr metadata,
        const api::ColumnGroupFile& file,
        const std::shared_ptr<arrow::Schema>& read_schema,
        const std::vector<std::string>& needed_columns,
        const std::string& predicate);
  };

  arrow::Status open() override;
  arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;
  arrow::Result<std::vector<uint64_t>> get_rg_column_memsz(int64_t index) const override;
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& index) override;
  arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(const std::vector<int>& indices) override;
  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& indices) override;
  arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(const uint64_t& start,
                                                                           const uint64_t& end) override;
  arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;
  std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  DeltaLakeFormatReader(MetaTrait::MetadataPtr metadata,
                        std::shared_ptr<DeltaFileMetadata> file,
                        std::shared_ptr<BlockingDeltaReadPlan> plan);
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> transform(const std::shared_ptr<arrow::RecordBatch>& batch,
                                                               uint64_t physical_start) const;

  MetaTrait::MetadataPtr metadata_;
  std::shared_ptr<DeltaFileMetadata> file_;
  std::shared_ptr<BlockingDeltaReadPlan> plan_;
  std::shared_ptr<arrow::Schema> output_schema_;
  std::shared_ptr<parquet::ParquetFormatReader> parquet_;
};

}  // namespace milvus_storage::deltalake
