#pragma once
#include "arrow/record_batch.h"
class Scanner {
  public:
  virtual std::shared_ptr<arrow::Table>
  Read() = 0;
  virtual void
  Close() = 0;
};