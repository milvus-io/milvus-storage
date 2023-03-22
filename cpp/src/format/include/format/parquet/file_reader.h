#pragma once
#include "arrow/filesystem/filesystem.h"
#include "arrow/table.h"
#include "file_scanner.h"
#include "format/reader.h"
#include "options.h"
#include "parquet/arrow/reader.h"
class ParquetFileReader : public Reader {
 public:
  ParquetFileReader(arrow::fs::FileSystem *fs, std::string &file_path, std::shared_ptr<ReadOption> &options);

  std::shared_ptr<Scanner> NewScanner() override;

  void Close() override {}

  std::shared_ptr<arrow::RecordBatch> ReadByOffsets(std::vector<int64_t> &offsets);
  std::shared_ptr<arrow::Table> ReadTable();

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOption> options_;
};