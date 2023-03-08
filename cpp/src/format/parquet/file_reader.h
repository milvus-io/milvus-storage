#pragma once

#include <arrow/record_batch.h>

#include <memory>

#include "../../options/options.h"
#include "arrow/filesystem/filesystem.h"
#include "parquet/arrow/reader.h"
class ParquetFileReader {
 public:
  ParquetFileReader(arrow::fs::FileSystem *fs, std::string &file_path,
                    std::shared_ptr<ReadOption> &options);
  arrow::RecordBatch Read();
  void Close();

 private:
  void initRecordReader();

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOption> options_;
};