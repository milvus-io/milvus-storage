#pragma once

#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include "file/fragment.h"
#include "storage/default_space.h"
#include "reader/multi_files_sequential_reader.h"

namespace milvus_storage {

class MultiFilesSequentialReader : public arrow::RecordBatchReader {
  public:
  MultiFilesSequentialReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                             const FragmentVector& fragments,
                             std::shared_ptr<arrow::Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> files_;

  size_t next_pos_;
  std::shared_ptr<arrow::RecordBatchReader> curr_reader_;
  std::shared_ptr<parquet::arrow::FileReader>
      holding_file_reader_;  // file reader have to outlive than record batch reader, so we hold here.

  friend FilterQueryRecordReader;
};
}  // namespace milvus_storage
