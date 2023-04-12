#include "file_scanner.h"
#include "parquet/arrow/reader.h"

namespace milvus_storage {

ParquetFileScanner::ParquetFileScanner(std::shared_ptr<parquet::arrow::FileReader> reader,
                                       std::shared_ptr<ReadOptions> option)
    : reader_(std::move(reader)), option_(std::move(option)) {}

Status ParquetFileScanner::Init() {
  auto metadata = reader_->parquet_reader()->metadata();
  std::vector<int> row_group_indices;
  std::vector<int> column_indices;

  for (const auto& column_name : option_->columns) {
    auto column_idx = metadata->schema()->ColumnIndex(column_name);
    column_indices.emplace_back(column_idx);
  }
  for (const auto& filter : option_->filters) {
    auto column_idx = metadata->schema()->ColumnIndex(filter->get_column_name());
    column_indices.emplace_back(column_idx);
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

  RETURN_ARROW_NOT_OK(reader_->GetRecordBatchReader(row_group_indices, column_indices, &record_reader_));
  return Status::OK();
}

Result<std::shared_ptr<arrow::Table>> ApplyFilter(std::shared_ptr<arrow::RecordBatch>& batch,
                                                  std::vector<Filter*>& filters) {
  filter_mask bitset;
  Filter::ApplyFilter(batch, filters, bitset);
  if (bitset.none()) {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto table, arrow::Table::FromRecordBatches({batch}));
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
    return Result<std::shared_ptr<arrow::Table>>(nullptr);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(filterd_batches));
  return res;
}

Result<std::shared_ptr<arrow::Table>> ParquetFileScanner::Read() {
  if (!record_reader_) {
    return Status::InternalStateError("Record reader is not initialized");
  }

  std::shared_ptr<arrow::Table> filtered_batch;
  do {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto rec_batch, record_reader_->Next());
    if (!rec_batch) {
      break;
    }
    ASSIGN_OR_RETURN_NOT_OK(filtered_batch, ApplyFilter(rec_batch, option_->filters));
  } while (!filtered_batch);

  if (!filtered_batch) {
    std::vector<int> column_indices;
    auto metadata = reader_->parquet_reader()->metadata();
    for (const auto& column_name : option_->columns) {
      auto column_idx = metadata->schema()->ColumnIndex(column_name);
      column_indices.emplace_back(column_idx);
    }

    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, filtered_batch->SelectColumns(column_indices));
    return res;
  }

  return Result<std::shared_ptr<arrow::Table>>(nullptr);
}
}  // namespace milvus_storage