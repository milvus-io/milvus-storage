#pragma once

#include "arrow/table.h"
#include "common/result.h"
namespace milvus_storage {

class Reader {
  public:
  virtual Status Init() = 0;

  virtual void Close() = 0;

  virtual Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) = 0;
};
}  // namespace milvus_storage