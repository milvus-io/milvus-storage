#include "format/parquet/file_scanner.h"

#include <memory>

#include "arrow/dataset/dataset.h"
#include "arrow/record_batch.h"
#include "exception.h"
#include "parquet/arrow/reader.h"

ParquetFileScanner::ParquetFileScanner(parquet::arrow::FileReader *reader, ReadOption *options) {
  InitRecordReader(reader, options);
}

void ParquetFileScanner::InitRecordReader(parquet::arrow::FileReader *reader, ReadOption *options) {
  auto metadata = reader->parquet_reader()->metadata();

  std::vector<int> row_group_indices;
  std::vector<int> column_indices;
  if (options->columns.size() == 0) {
    for (int i = 0; i < metadata->num_columns(); ++i) {
      column_indices.emplace_back(i);
    }
  } else {
    for (const auto &column_name : options->columns) {
      auto column_idx = metadata->schema()->ColumnIndex(column_name);
      column_indices.emplace_back(column_idx);
    }
  }

  for (int i = 0; i < metadata->num_row_groups(); ++i) {
    auto row_group_metadata = metadata->RowGroup(i);
    bool can_ignored = false;

    for (const auto &filter : options->filters) {
      auto column_idx = metadata->schema()->ColumnIndex(filter->get_column_name());
      auto column_meta = row_group_metadata->ColumnChunk(column_idx);
      auto stats = column_meta->statistics();

      if (stats == nullptr || !stats->HasMinMax()) {
        continue;
      }
      if (filter->CheckStatistics(stats.get())) {
        can_ignored = true;
        break;
      }
    }
    if (!can_ignored) {
      row_group_indices.emplace_back(i);
    }
  }

  auto status = reader->GetRecordBatchReader(row_group_indices, column_indices, &record_reader_);
  if (!status.ok()) {
    throw StorageException("get record reader failed");
  }
}

std::shared_ptr<arrow::RecordBatch> ParquetFileScanner::Read() {
  if (record_reader_ == nullptr) {
    throw StorageException("record reader is null");
  }
  auto res = record_reader_->Next();
  PARQUET_THROW_NOT_OK(res);
  return res.ValueOrDie();
}