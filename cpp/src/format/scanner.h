#pragma once
#include "parquet/arrow/reader.h"
class Scanner {
 public:
  virtual std::shared_ptr<arrow::RecordBatch> Read() = 0;
  virtual void Close() = 0;
};