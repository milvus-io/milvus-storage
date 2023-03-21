#pragma once

#include <memory>

#include "arrow/record_batch.h"
#include "scanner.h"
class Reader {
 public:
  virtual std::shared_ptr<Scanner> NewScanner() = 0;
  virtual void Close() = 0;
  virtual std::shared_ptr<arrow::RecordBatch> ReadByOffsets(std::vector<int64_t> &offsets) = 0;

  template <typename T>
  std::shared_ptr<arrow::RecordBatch> ReadByPks(std::vector<T>){};
};