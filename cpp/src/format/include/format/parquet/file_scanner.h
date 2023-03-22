#pragma once

#include <arrow/record_batch.h>

#include <memory>

#include "arrow/filesystem/filesystem.h"
#include "format/scanner.h"
#include "options.h"
#include "parquet/arrow/reader.h"
class ParquetFileScanner : public Scanner {
 public:
  ParquetFileScanner(parquet::arrow::FileReader *reader, std::shared_ptr<ReadOption> option);
  std::shared_ptr<arrow::Table> Read() override;
  void Close() override { record_reader_->Close(); }

 private:
  void InitRecordReader(parquet::arrow::FileReader *);

 private:
  std::shared_ptr<arrow::RecordBatchReader> record_reader_;
  std::shared_ptr<ReadOption> option_;
};