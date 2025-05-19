

#pragma once

#include "arrow/table.h"
#include "milvus-storage/common/result.h"
namespace milvus_storage {

class Reader {
  public:
  virtual void Close() = 0;

  virtual Result<std::shared_ptr<arrow::Table>> ReadByOffsets(std::vector<int64_t>& offsets) = 0;

  virtual ~Reader() = default;
};
}  // namespace milvus_storage