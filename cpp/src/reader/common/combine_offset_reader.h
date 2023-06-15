#pragma once

#include <memory>
#include "arrow/record_batch.h"
#include "format/parquet/file_reader.h"
#include "storage/schema.h"
namespace milvus_storage {

// CombineOffsetReader reads records from a reader and fetches corresponding records
// of another file and combines them together.
class CombineOffsetReader : public arrow::RecordBatchReader {
  public:
  static Result<std::shared_ptr<CombineOffsetReader>> Make(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                                                           std::shared_ptr<ParquetFileReader> vector_reader,
                                                           std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  CombineOffsetReader(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                      std::shared_ptr<ParquetFileReader> vector_reader,
                      std::shared_ptr<Schema> schema)
      : scalar_reader_(std::move(scalar_reader)),
        vector_reader_(std::move(vector_reader)),
        schema_(std::move(schema)) {}

  private:
  std::shared_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::shared_ptr<ParquetFileReader> vector_reader_;
  std::shared_ptr<Schema> schema_;
};
}  // namespace milvus_storage
