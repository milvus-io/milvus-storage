#pragma once

#include <arrow/record_batch.h>

#include <memory>

#include "arrow/filesystem/filesystem.h"
#include "format/scanner.h"
#include "options.h"
#include "parquet/arrow/reader.h"
class ParquetFileScanner : public Scanner {
 public:
  ParquetFileScanner(parquet::arrow::FileReader *reader, ReadOption *options);
  std::shared_ptr<arrow::RecordBatch> Read() override;
  void Close() override { record_reader_->Close(); }

 private:
  void InitRecordReader(parquet::arrow::FileReader *, ReadOption *);

 private:
  std::shared_ptr<arrow::RecordBatchReader> record_reader_;
};