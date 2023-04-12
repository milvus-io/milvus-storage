#pragma once
#include "format/scanner.h"
#include "storage/options.h"

namespace milvus_storage {

class ParquetFileScanner : public Scanner {
  public:
  ParquetFileScanner(std::shared_ptr<parquet::arrow::FileReader> reader, std::shared_ptr<ReadOptions> option);

  Status Init();
  Result<std::shared_ptr<arrow::Table>> Read() override;

  void Close() override { record_reader_->Close(); }

  private:
  std::shared_ptr<arrow::RecordBatchReader> record_reader_;
  std::shared_ptr<ReadOptions> option_;
  // reader_ must have a longer lifetime than record_reader_
  std::shared_ptr<parquet::arrow::FileReader> reader_;
};
}  // namespace milvus_storage