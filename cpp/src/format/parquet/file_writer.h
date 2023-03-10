#pragma once
#include <arrow/type_fwd.h>

#include "../../options/options.h"
#include "../writer.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/record_batch.h"
#include "parquet/file_writer.h"

class ParquetFileWriter : public FileWriter {
 public:
  ParquetFileWriter(arrow::Schema *schema, arrow::fs::FileSystem *fs,
                    std::string &file_path);
  void Write(arrow::RecordBatch *record) override;
  int64_t count() override;
  void Close() override;

 private:
  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  int64_t count_ = 0;
};
