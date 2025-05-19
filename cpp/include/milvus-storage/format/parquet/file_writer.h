

#pragma once

#include <memory>
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/writer.h"
#include "parquet/arrow/writer.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include <arrow/util/key_value_metadata.h>
#include "milvus-storage/common/config.h"

namespace milvus_storage {

class ParquetFileWriter : public FileWriter {
  public:
  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    const std::string& file_path,
                    const StorageConfig& storage_config,
                    std::shared_ptr<parquet::WriterProperties> writer_props = parquet::default_writer_properties());

  Status Init() override;

  Status Write(const arrow::RecordBatch& record) override;

  Status WriteTable(const arrow::Table& table) override;

  Status WriteRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                            const std::vector<size_t>& batch_memory_sizes);

  void AppendKVMetadata(const std::string& key, const std::string& value);

  int64_t count() override;

  Status Close() override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;
  const StorageConfig& storage_config_;

  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  std::shared_ptr<arrow::KeyValueMetadata> kv_metadata_;
  int64_t count_ = 0;
  RowGroupMetadataVector row_group_metadata_;
  std::shared_ptr<parquet::WriterProperties> writer_props_;
};
}  // namespace milvus_storage
