#pragma once
#include <memory>
#include <utility>

#include "options.h"
#include "arrow/record_batch.h"
namespace milvus_storage {

class Space {
  public:
  explicit Space(std::shared_ptr<SpaceOptions> options) : options_(std::move(options)){};
  virtual void
  Write(arrow::RecordBatchReader* reader, WriteOption* option) = 0;
  virtual std::unique_ptr<arrow::RecordBatchReader>
  Read(std::shared_ptr<ReadOptions> option) = 0;
  virtual void
  Delete(arrow::RecordBatchReader* reader) = 0;

  protected:
  std::shared_ptr<SpaceOptions> options_;
};
}  // namespace milvus_storage