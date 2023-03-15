#pragma once
#include "arrow/filesystem/filesystem.h"
#include "options.h"
#include "parquet-format/file_scanner.h"
#include "parquet/arrow/reader.h"
#include "reader.h"
class ParquetFileReader : public Reader {
 public:
  ParquetFileReader(arrow::fs::FileSystem *fs, std::string &file_path,
                    std::shared_ptr<ReadOption> &options);

  std::shared_ptr<Scanner> NewScanner() override;

  void Close() override {}

  std::shared_ptr<arrow::RecordBatch> ReadByOffsets(
      std::vector<int64_t> &offsets);

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOption> options_;
};