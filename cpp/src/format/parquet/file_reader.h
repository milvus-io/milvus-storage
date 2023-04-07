#pragma once
#include "arrow/filesystem/filesystem.h"
#include "arrow/table.h"
#include "format/parquet/file_scanner.h"
#include "format/reader.h"
#include "parquet/arrow/reader.h"
#include "storage/options.h"
namespace milvus_storage {

class ParquetFileReader : public Reader {
  public:
  ParquetFileReader(arrow::fs::FileSystem* fs, std::string& file_path, std::shared_ptr<ReadOptions>& options);

  std::shared_ptr<Scanner>
  NewScanner() override;

  void
  Close() override {
  }

  std::shared_ptr<arrow::Table>
  ReadByOffsets(std::vector<int64_t>& offsets);

  private:
  std::shared_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOptions> options_;
};
}  // namespace milvus_storage