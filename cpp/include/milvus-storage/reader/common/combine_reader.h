

#pragma once

#include <memory>
#include "arrow/record_batch.h"
#include "milvus-storage/storage/schema.h"

namespace milvus_storage {

// CombineReader merges scalar fields and vector fields to an entire record.
class CombineReader : public arrow::RecordBatchReader {
  public:
  static std::unique_ptr<CombineReader> Make(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                                             std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                                             std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  CombineReader(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                std::shared_ptr<Schema> schema)
      : scalar_reader_(std::move(scalar_reader)),
        vector_reader_(std::move(vector_reader)),
        schema_(std::move(schema)) {}

  private:
  std::unique_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::unique_ptr<arrow::RecordBatchReader> vector_reader_;
  std::shared_ptr<Schema> schema_;
};
}  // namespace milvus_storage
