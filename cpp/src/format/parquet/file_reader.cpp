#include "parquet-format/file_reader.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>

#include <memory>

#include "exception.h"

std::shared_ptr<arrow::RecordBatch> BuildRecordBatch(
    std::shared_ptr<arrow::Schema> &schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>> &batchs) {
  std::unique_ptr<arrow::RecordBatchBuilder> builder =
      arrow::RecordBatchBuilder::Make(schema, arrow::default_memory_pool())
          .ValueOrDie();
  for (const auto &batch : batchs) {
    for (int i = 0; i < batch->num_columns(); ++i) {
      auto status = builder->GetField(i)->AppendArraySlice(
          *batch->column(i)->data().get(), 0, 1);
      if (!status.ok()) {
        throw StorageException("append array slice failed");
      }
    }
  }

  return builder->Flush().ValueOrDie();
}

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

std::shared_ptr<arrow::RecordBatch> ParquetFileReader::ReadByOffsets(
    std::vector<int64_t> &offsets) {
  std::sort(offsets.begin(), offsets.end());

  std::vector<std::shared_ptr<arrow::RecordBatch>> batchs;
  auto num_row_groups = reader_->parquet_reader()->metadata()->num_row_groups();
  int offset_idx = 0;
  int64_t skipped = 0;
  for (int i = 0; i < num_row_groups; ++i) {
    // skip rowgroups before the first record in offsets
    auto row_group = reader_->parquet_reader()->RowGroup(i);
    auto next_skipped = row_group->metadata()->num_rows() + skipped;
    if (next_skipped < offsets[offset_idx]) {
      offset_idx++;
      skipped = next_skipped;
      continue;
    }

    // read records in the row group
    std::unique_ptr<arrow::RecordBatchReader> batch_reader;
    auto status = reader_->GetRecordBatchReader({i}, &batch_reader);
    if (!status.ok()) {
      throw StorageException("get record reader failed");
    }
    std::shared_ptr<arrow::RecordBatch> current_batch;
    int64_t current_skipped_row_group = 0;
    while (offset_idx < offsets.size()) {
      if (next_skipped < offsets[offset_idx]) {
        // records in offsets are not in this row group
        break;
      }

      auto row_group_offset = offsets[offset_idx] - skipped;

      if (current_batch == nullptr) {
        auto status = batch_reader->ReadNext(&current_batch);
        if (!status.ok()) {
          throw StorageException("read batch failed");
        }
      }
      while (current_batch->num_rows() + current_skipped_row_group <
             row_group_offset) {
        // skip batch before this record
        current_skipped_row_group += current_batch->num_rows();
        auto status = batch_reader->ReadNext(&current_batch);
        if (!status.ok()) {
          throw StorageException("read batch failed");
        }
      }

      auto offset_batch = row_group_offset - current_skipped_row_group;
      auto rec = current_batch->Slice(offset_batch, 1);
      offset_idx++;
      batchs.emplace_back(rec);
    }

    skipped = next_skipped;
  }

  std::shared_ptr<arrow::Schema> schema;
  auto status = reader_->GetSchema(&schema);
  if (!status.ok()) {
    throw StorageException("read batch failed");
  }
  return BuildRecordBatch(schema, batchs);
}
