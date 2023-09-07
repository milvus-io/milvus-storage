#pragma once
#include "arrow/record_batch.h"
#include "common/status.h"
namespace milvus_storage {

class FileWriter {
  public:
  virtual Status Init() = 0;

  virtual Status Write(arrow::RecordBatch* record) = 0;

  virtual int64_t count() = 0;

  virtual Status Close() = 0;
};
}  // namespace milvus_storage