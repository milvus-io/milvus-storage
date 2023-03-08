#pragma once
#include <memory>
#include <utility>

#include "../options/options.h"
#include "arrow/record_batch.h"

class Space {
 public:
  explicit Space(std::shared_ptr<SpaceOption> options)
      : options(std::move(options)){};
  virtual void Write(arrow::RecordBatchReader *reader, WriteOption *option);
  virtual std::shared_ptr<arrow::RecordBatch> Read(
      std::shared_ptr<ReadOption> option);
  virtual void DeleteByPks(arrow::RecordBatchReader *reader);

 private:
  std::shared_ptr<SpaceOption> options;
};
