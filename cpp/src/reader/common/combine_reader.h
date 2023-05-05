#pragma once

#include <memory>
#include "arrow/record_batch.h"
#include "common/result.h"
#include "storage/schema.h"

namespace milvus_storage {
class CombineReader : public arrow::RecordBatchReader {
  public:
  static Result<std::shared_ptr<CombineReader>> Make(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                                                     std::shared_ptr<arrow::RecordBatchReader> vector_reader,
                                                     std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  CombineReader(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                std::shared_ptr<arrow::RecordBatchReader> vector_reader,
                std::shared_ptr<Schema> schema)
      : scalar_reader_(std::move(scalar_reader)),
        vector_reader_(std::move(vector_reader)),
        schema_(std::move(schema)) {}
  std::shared_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::shared_ptr<arrow::RecordBatchReader> vector_reader_;
  std::shared_ptr<Schema> schema_;
};
}  // namespace milvus_storage