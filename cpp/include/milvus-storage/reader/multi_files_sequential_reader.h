

#pragma once

#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include "milvus-storage/file/fragment.h"
#include "milvus-storage/storage/space.h"

namespace milvus_storage {

class MultiFilesSequentialReader : public arrow::RecordBatchReader {
  public:
  MultiFilesSequentialReader(arrow::fs::FileSystem& fs,
                             const FragmentVector& fragments,
                             std::shared_ptr<arrow::Schema> schema,
                             const SchemaOptions& schema_options,
                             const ReadOptions& options);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  arrow::fs::FileSystem& fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> files_;

  size_t next_pos_ = 0;
  std::unique_ptr<arrow::RecordBatchReader> curr_reader_;
  std::unique_ptr<parquet::arrow::FileReader>
      holding_file_reader_;  // file reader have to outlive than record batch reader, so we hold here.
  const ReadOptions options_;
  const SchemaOptions schema_options_;

  friend FilterQueryRecordReader;
};
}  // namespace milvus_storage
