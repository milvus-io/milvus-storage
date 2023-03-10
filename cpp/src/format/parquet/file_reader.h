#pragma once
#include "../../options/options.h"
#include "arrow/filesystem/filesystem.h"
#include "file_scanner.h"
class ParquetFileReader {
 public:
  ParquetFileReader(arrow::fs::FileSystem *fs, std::string &file_path,
                    std::shared_ptr<ReadOption> &options);
  std::unique_ptr<Scanner> NewScanner();
  void Close() {}

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOption> options_;
};