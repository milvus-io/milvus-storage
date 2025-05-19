

#pragma once

#include <arrow/type.h>
#include "arrow/record_batch.h"
#include <memory>
#include "milvus-storage/storage/options.h"
#include "milvus-storage/common/result.h"
namespace milvus_storage {
class ProjectionReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static Result<std::unique_ptr<arrow::RecordBatchReader>> Make(std::shared_ptr<arrow::Schema> schema,
                                                                std::unique_ptr<arrow::RecordBatchReader> reader,
                                                                const ReadOptions& options);

  ProjectionReader(std::shared_ptr<arrow::Schema> schema,
                   std::unique_ptr<arrow::RecordBatchReader> reader,
                   const ReadOptions& options);

  private:
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  const ReadOptions options_;
  std::shared_ptr<arrow::Schema> schema_;
};
}  // namespace milvus_storage
