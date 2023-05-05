#pragma once
#include "arrow/filesystem/filesystem.h"
#include "format/reader.h"
#include "parquet/arrow/reader.h"
#include "storage/options.h"
namespace milvus_storage {

class ParquetFileReader : public Reader {
  public:
  ParquetFileReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::string& file_path,
                    std::shared_ptr<ReadOptions>& options);

  void Close() override {}

  Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string file_path_;

  std::shared_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<ReadOptions> options_;
};
}  // namespace milvus_storage