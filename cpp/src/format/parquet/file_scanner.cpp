#include "format/parquet/file_scanner.h"
#include <arrow/type_fwd.h>
#include <parquet/exception.h>

#include <memory>
#include <utility>

#include "arrow/dataset/dataset.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "common/exception.h"
#include "parquet/arrow/reader.h"

ParquetFileScanner::ParquetFileScanner(std::shared_ptr<parquet::arrow::FileReader> reader,
                                       std::shared_ptr<ReadOption> option)
    : reader_(std::move(reader)), option_(std::move(option)) {
  auto metadata = reader->parquet_reader()->metadata();
  std::vector<int> row_group_indices;
  std::vector<int> column_indices;
  if (option_->columns.size() == 0) {
    for (int i = 0; i < metadata->num_columns(); ++i) {
      column_indices.emplace_back(i);
    }
  } else {
    for (const auto& column_name : option_->columns) {
      auto column_idx = metadata->schema()->ColumnIndex(column_name);
      column_indices.emplace_back(column_idx);
    }
  }

  for (int i = 0; i < metadata->num_row_groups(); ++i) {
    auto row_group_metadata = metadata->RowGroup(i);
    bool can_ignored = false;

    for (const auto& filter : option_->filters) {
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

std::shared_ptr<arrow::Table>
ApplyFilter(std::shared_ptr<arrow::RecordBatch>& batch, std::vector<Filter*>& filters) {
  filter_mask bitset;
  Filter::ApplyFilter(batch, filters, bitset);
  if (bitset.none()) {
    PARQUET_ASSIGN_OR_THROW(auto table, arrow::Table::FromRecordBatches({batch}));
    return table;
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> filterd_batches;
  int start_idx = 0, end_idx = 0;
  int64_t num_rows = batch->num_rows();
  while (end_idx < num_rows) {
    while (end_idx < num_rows && bitset.test(end_idx)) {
      end_idx++;
      start_idx++;
    }
    while (end_idx < num_rows && !bitset.test(end_idx)) {
      end_idx++;
    }

    if (start_idx >= num_rows) {
      break;
    }
    filterd_batches.emplace_back(batch->Slice(start_idx, end_idx - start_idx));
    start_idx = end_idx;
  }

  if (filterd_batches.empty()) {
    return nullptr;
  }

  PARQUET_ASSIGN_OR_THROW(auto res, arrow::Table::FromRecordBatches(filterd_batches));
  return res;
}

std::shared_ptr<arrow::Table>
ParquetFileScanner::Read() {
  if (!record_reader_) {
    throw StorageException("record reader is null");
  }

  std::shared_ptr<arrow::Table> res;
  do {
    PARQUET_ASSIGN_OR_THROW(auto rec_batch, record_reader_->Next());
    if (!rec_batch) {
      break;
    }
    res = ApplyFilter(rec_batch, option_->filters);
  } while (!res);
  return res;
}
