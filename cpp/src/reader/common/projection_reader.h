#pragma once

#include <arrow/type.h>
#include "arrow/record_batch.h"
#include <memory>
#include "storage/options.h"
#include "common/result.h"
namespace milvus_storage {
class ProjectionReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static Result<std::shared_ptr<arrow::RecordBatchReader>> Make(std::shared_ptr<arrow::Schema> schema,
                                                                std ::shared_ptr<arrow::RecordBatchReader> reader,
                                                                std::shared_ptr<ReadOptions> options);

  ProjectionReader(std::shared_ptr<arrow::Schema> schema,
                   std ::shared_ptr<arrow::RecordBatchReader> reader,
                   std::shared_ptr<ReadOptions> options);

  private:
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<ReadOptions> options_;
  std::shared_ptr<arrow::Schema> schema_;
};
}  // namespace milvus_storage
