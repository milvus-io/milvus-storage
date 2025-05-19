

#pragma once

#include <arrow/record_batch.h>
#include <parquet/properties.h>
#include <memory>
#include "milvus-storage/format/parquet/file_writer.h"
#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/packed/column_group.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/common/metadata.h"

namespace milvus_storage {

class ColumnGroupWriter {
  public:
  ColumnGroupWriter(GroupId group_id,
                    std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    const std::string& file_path,
                    const StorageConfig& storage_config,
                    const std::vector<int>& origin_column_indices,
                    std::shared_ptr<parquet::WriterProperties> writer_props);

  Status Init();
  Status Write(const std::shared_ptr<arrow::RecordBatch>& record);
  Status WriteGroupFieldIDList(const GroupFieldIDList& list);
  Status Flush();
  Status Close();
  GroupId Group_id() const;

  private:
  bool finished_;
  GroupId group_id_;
  ParquetFileWriter writer_;
  ColumnGroup column_group_;
  int flushed_batches_;
  int flushed_count_;
  int64_t flushed_rows_;
};

}  // namespace milvus_storage
