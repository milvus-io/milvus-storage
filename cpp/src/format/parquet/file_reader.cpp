#include "file_reader.h"

#include <arrow/filesystem/type_fwd.h>

#include <memory>

#include "../../exception.h"
#include "arrow/dataset/dataset.h"
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"

ParquetFileReader::ParquetFileReader(arrow::fs::FileSystem *fs,
                                     std::string &file_path,
                                     std::shared_ptr<ReadOption> &options) {
  auto res = fs->OpenInputFile(file_path);
  if (!res.ok()) {
    throw StorageException("open file failed");
  }

  auto status = parquet::arrow::OpenFile(
      res.ValueOrDie(), arrow::default_memory_pool(), &reader_);
  if (!status.ok()) {
    throw StorageException("open file reader failed");
  }

  options_ = options;
}

void ParquetFileReader::initRecordReader() {
  auto parquet_reader = reader_->parquet_reader();
  auto metadata = parquet_reader->metadata();
  auto row_group_num = metadata->num_row_groups();
  auto schema = metadata->schema();
  for (int i = 0; i < row_group_num; ++i) {
    auto row_group_metadata = metadata->RowGroup(i);
    for (const auto &col : options_->columns) {
      auto col_idx = schema->ColumnIndex(col);
      auto col_meta = row_group_metadata->ColumnChunk(col_idx);
      col_meta->statistics()
    }
  }
}

arrow::RecordBatch ParquetFileReader::Read() {}