#pragma once
#include <memory>
#include <utility>

#include "../options/options.h"
#include "arrow/record_batch.h"

class Space {
 public:
  explicit Space(std::shared_ptr<SpaceOption> options)
      : options_(std::move(options)){};
  virtual void Write(arrow::RecordBatchReader *reader, WriteOption *option) = 0;
  virtual std::shared_ptr<arrow::RecordBatch> Read(
      std::shared_ptr<ReadOption> option) = 0;
  virtual void DeleteByPks(arrow::RecordBatchReader *reader) = 0;

 protected:
  std::shared_ptr<SpaceOption> options_;
};
