#include "format/parquet/file_reader.h"

#include "arrow/table.h"
#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <parquet/exception.h>
#include <vector>

#include "common/exception.h"

ParquetFileReader::ParquetFileReader(arrow::fs::FileSystem *fs,
                                     std::string &file_path,
                                     std::shared_ptr<ReadOption> &options)
    : options_(options) {
  auto res = fs->OpenInputFile(file_path);
  if (!res.ok()) {
    throw StorageException("open file failed");
  }
  auto status = parquet::arrow::OpenFile(
      res.ValueOrDie(), arrow::default_memory_pool(), &reader_);
  if (!status.ok()) {
    throw StorageException("open file reader failed");
  }
}

std::shared_ptr<Scanner> ParquetFileReader::NewScanner() {
  return std::make_shared<ParquetFileScanner>(reader_.get(), options_.get());
}

std::shared_ptr<arrow::RecordBatch>
GetRecordAtOffset(arrow::RecordBatchReader *reader, int64_t offset) {
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
ApplyFilter(const std::vector<std::shared_ptr<arrow::RecordBatch>> &batches,
            std::vector<Filter *> &filters) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> filterd_batches;
  for (const auto &batch : batches) {
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

  PARQUET_ASSIGN_OR_THROW(auto res,
                          arrow::Table::FromRecordBatches(filterd_batches));
  return res;
}
