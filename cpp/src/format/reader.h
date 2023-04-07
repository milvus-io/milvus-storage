#pragma once

#include <memory>

#include "arrow/record_batch.h"
#include "scanner.h"
namespace milvus_storage {

class Reader {
  public:
  virtual std::shared_ptr<Scanner>
  NewScanner() = 0;
  virtual void
  Close() = 0;
  virtual std::shared_ptr<arrow::Table>
  ReadByOffsets(std::vector<int64_t>& offsets) = 0;
};
}  // namespace milvus_storage