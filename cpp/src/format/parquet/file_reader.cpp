#include "format/parquet/file_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <parquet/type_fwd.h>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>
#ijjjjnclude "arrow/table.h"
#include "common/macro.h"

namespace milvus_storage {

ParquetFileReader::ParquetFileReader(std::shared_ptr<parquet::arrow::FileReader> reader,
                                     std::shared_ptr<ReadOptions>& options)
    : reader_(std::move(reader)), options_(options) {}

Result<std::shared_ptr<arrow::RecordBatch>> GetRecordAtOffset(arrow::RecordBatchReader* reader, int64_t offset) {
  int64_t skipped = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  do {
    RETURN_ARROW_NOT_OK(reader->ReadNext(&batch));
    skipped += batch->num_rows();
  } while (skipped < offset);

  auto offset_batch = offset - skipped + batch->num_rows();
  // zero-copy slice
  return batch->Slice(offset_batch, 1);
}

// TODO: support projection
Result<std::shared_ptr<arrow::Table>> ParquetFileReader::ReadByOffsets(std::vector<int64_t>& offsets) {
  std::sort(offsets.begin(), offsets.end());

  auto num_row_groups = reader_->parquet_reader()->metadata()->num_row_groups();
  int current_row_group_idx = 0;
  int64_t total_skipped = 0;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::unique_ptr<arrow::RecordBatchReader> current_row_group_reader;

  for (auto& offset : offsets) {
    // skip row groups
    // TODO: to make read more efficient, we should find offsets belonged to a row group and read together.
    while (current_row_group_idx < num_row_groups) {
      auto row_group_meta = reader_->parquet_reader()->metadata()->RowGroup(current_row_group_idx);
      auto row_group_num_rows = row_group_meta->num_rows();
      if (row_group_num_rows + total_skipped > offset) {
        break;
      }
      current_row_group_idx++;
      total_skipped += row_group_num_rows;
      current_row_group_reader = nullptr;
    }

    if (current_row_group_idx >= num_row_groups) {
      break;
    }

    if (current_row_group_reader == nullptr) {
      RETURN_ARROW_NOT_OK(reader_->GetRecordBatchReader({current_row_group_idx}, &current_row_group_reader));
    }

    auto row_group_offset = offset - total_skipped;
    ASSIGN_OR_RETURN_NOT_OK(auto batch, GetRecordAtOffset(current_row_group_reader.get(), row_group_offset))
    batches.push_back(batch);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(batches));
  return res;
}
}  // namespace milvus_storage
