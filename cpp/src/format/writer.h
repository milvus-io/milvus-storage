#pragma once
#include "options.h"
#include "arrow/record_batch.h"

class FileWriter {
 public:
  virtual void Write(arrow::RecordBatch *record) = 0;
  virtual int64_t count() = 0;
  virtual void Close() = 0;
};