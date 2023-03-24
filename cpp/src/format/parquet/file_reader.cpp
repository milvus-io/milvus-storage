#include "format/parquet/file_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <parquet/exception.h>
#include <parquet/type_fwd.h>
#include <memory>
#include <vector>
#include "arrow/table.h"

#include "common/exception.h"

ParquetFileReader::ParquetFileReader(arrow::fs::FileSystem* fs,
                                     std::string& file_path,
                                     std::shared_ptr<ReadOption>& options)
    : options_(options) {
  auto res = fs->OpenInputFile(file_path);
  if (!res.ok()) {
    throw StorageException("open file failed");
  }
  std::unique_ptr<parquet::arrow::FileReader> file_reader;
  auto status = parquet::arrow::OpenFile(res.ValueOrDie(), arrow::default_memory_pool(), &file_reader);
  reader_ = std::move(file_reader);
  if (!status.ok()) {
    throw StorageException("open file reader failed");
  }
}

std::shared_ptr<Scanner>
ParquetFileReader::NewScanner() {
  return std::make_shared<ParquetFileScanner>(reader_, options_);
}

std::shared_ptr<arrow::RecordBatch>
GetRecordAtOffset(arrow::RecordBatchReader* reader, int64_t offset) {
  int64_t skipped = 0;
  std::shared_ptr<arrow::RecordBatch> batch;

  do {
    PARQUET_THROW_NOT_OK(reader->ReadNext(&batch));
    skipped += batch->num_rows();
  } while (skipped < offset);

  auto offset_batch = offset - skipped + batch->num_rows();
  return batch->Slice(offset_batch, 1);
}

std::shared_ptr<arrow::Table>
ApplyFilter(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, std::vector<Filter*>& filters) {
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
    return nullptr;
  }

  PARQUET_ASSIGN_OR_THROW(auto res, arrow::Table::FromRecordBatches(filterd_batches));
  return res;
}

// TODO: support projection
std::shared_ptr<arrow::Table>
ParquetFileReader::ReadByOffsets(std::vector<int64_t>& offsets) {
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
      auto status = reader_->GetRecordBatchReader({current_row_group_idx}, &current_row_group_reader);
      if (!status.ok()) {
        throw StorageException("get record reader failed");
      }
    }

    auto row_group_offset = offsets[i] - total_skipped;
    std::shared_ptr<arrow::RecordBatch> batch = GetRecordAtOffset(current_row_group_reader.get(), row_group_offset);
    batches.push_back(batch);
  }

  return ApplyFilter(batches, options_->filters);
}