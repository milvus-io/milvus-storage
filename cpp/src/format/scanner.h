#pragma once
#include "arrow/table.h"
#include "common/result.h"
namespace milvus_storage {

class Scanner {
  public:
  virtual Result<std::shared_ptr<arrow::Table>> Read() = 0;
  virtual void Close() = 0;
};
}  // namespace milvus_storage