#pragma once
#include "arrow/record_batch.h"
namespace milvus_storage {

class Scanner {
  public:
  virtual std::shared_ptr<arrow::Table>
  Read() = 0;
  virtual void
  Close() = 0;
};
}  // namespace milvus_storage