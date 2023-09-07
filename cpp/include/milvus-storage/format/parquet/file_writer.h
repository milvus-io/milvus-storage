#pragma once

#include "arrow/filesystem/filesystem.h"
#include "format/writer.h"
#include "parquet/arrow/writer.h"
namespace milvus_storage {

class ParquetFileWriter : public FileWriter {
  public:
  ParquetFileWriter(std::shared_ptr<arrow::Schema> schema,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    std::string& file_path);

  Status Init() override;

  Status Write(arrow::RecordBatch* record) override;

  int64_t count() override;

  Status Close() override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string file_path_;

  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  int64_t count_ = 0;
};
}  // namespace milvus_storage