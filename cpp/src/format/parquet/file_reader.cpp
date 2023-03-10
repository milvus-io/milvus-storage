#include "file_reader.h"

#include <memory>

#include "../../exception.h"

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

std::unique_ptr<Scanner> ParquetFileReader::NewScanner() {
  return std::make_unique<ParquetFileScanner>(reader_.get(), options_.get());
}
