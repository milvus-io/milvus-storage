#include "format/parquet/file_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <parquet/type_fwd.h>
#include <memory>
#include <utility>
#include <vector>
#include "arrow/table.h"
#include "common/macro.h"

namespace milvus_storage {

ParquetFileReader::ParquetFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                     std::string& file_path,
                                     std::shared_ptr<ReadOptions>& options)
    : fs_(std::move(fs)), file_path_(file_path), options_(options) {}

Status ParquetFileReader::Init() {
  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto file, fs_->OpenInputFile(file_path_));

  std::unique_ptr<parquet::arrow::FileReader> file_reader;
  RETURN_ARROW_NOT_OK(parquet::arrow::OpenFile(file, arrow::default_memory_pool(), &file_reader));
  reader_ = std::move(file_reader);

  return Status::OK();
}

std::shared_ptr<ParquetFileScanner> ParquetFileReader::NewScanner() {
  return std::make_shared<ParquetFileScanner>(reader_, options_);
}
Result<std::shared_ptr<arrow::RecordBatch>> GetRecordAtOffset(arrow::RecordBatchReader* reader, int64_t offset) {
  int64_t skipped = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  do {
    RETURN_ARROW_NOT_OK(reader->ReadNext(&batch));
    skipped += batch->num_rows();
  } while (skipped < offset);

  auto offset_batch = offset - skipped + batch->num_rows();
  return batch->Slice(offset_batch, 1);
}
Result<std::shared_ptr<arrow::Table>> ApplyFilter(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                                                  std::vector<Filter*>& filters) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> filterd_batches;
  for (const auto& batch : batches) {
    filter_mask bitset;
    Filter::ApplyFilter(batch, filters, bitset);
    if (bitset.test(0)) {
      continue;
    }
    filterd_batches.emplace_back(batch);
  }
  if (filterd_batches.empty()) {
    return Result<std::shared_ptr<arrow::Table>>(nullptr);
  }

  ASSIGN_OR_RETURN_ARROW_NOT_OK(auto res, arrow::Table::FromRecordBatches(filterd_batches));
  return res;
}

// TODO: support projection
Result<std::shared_ptr<arrow::Table>> ParquetFileReader::ReadByOffsets(std::vector<int64_t>& offsets) {
  std::sort(offsets.begin(), offsets.end());
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  auto num_row_groups = reader_->parquet_reader()->metadata()->num_row_groups();
  int current_row_group_idx = 0;
  int64_t total_skipped = 0;
  std::unique_ptr<arrow::RecordBatchReader> current_row_group_reader;
  for (int i = 0; i < offsets.size(); ++i) {
    // skip row groups
    while (current_row_group_idx < num_row_groups) {
      auto row_group_meta = reader_->parquet_reader()->metadata()->RowGroup(current_row_group_idx);
      auto row_group_num_rows = row_group_meta->num_rows();
      if (row_group_num_rows + total_skipped > offsets[i]) {
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

    auto row_group_offset = offsets[i] - total_skipped;
    ASSIGN_OR_RETURN_NOT_OK(auto batch, GetRecordAtOffset(current_row_group_reader.get(), row_group_offset))
    batches.push_back(batch);
  }

  return ApplyFilter(batches, options_->filters);
}
}  // namespace milvus_storage