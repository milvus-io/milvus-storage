#pragma once

#include <arrow/record_batch.h>
#include <parquet/type_fwd.h>

#include <memory>

#include "arrow/filesystem/filesystem.h"
#include "format/scanner.h"
#include "parquet/arrow/reader.h"
#include "storage/options.h"
class ParquetFileScanner : public Scanner {
  public:
  ParquetFileScanner(std::shared_ptr<parquet::arrow::FileReader> reader, std::shared_ptr<ReadOptions> option);
  std::shared_ptr<arrow::Table>
  Read() override;
  void
  Close() override {
    record_reader_->Close();
  }

  private:
  std::shared_ptr<arrow::RecordBatchReader> record_reader_;
  std::shared_ptr<ReadOptions> option_;
  // reader_ must have a longer lifetime than record_reader_
  std::shared_ptr<parquet::arrow::FileReader> reader_;
};
