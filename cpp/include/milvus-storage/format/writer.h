

#pragma once
#include "arrow/record_batch.h"
#include "milvus-storage/common/status.h"
namespace milvus_storage {

class FileWriter {
  public:
  virtual Status Init() = 0;

  virtual Status Write(const arrow::RecordBatch& record) = 0;

  virtual Status WriteTable(const arrow::Table& table) = 0;

  virtual int64_t count() = 0;

  virtual Status Close() = 0;

  virtual ~FileWriter() = default;
};
}  // namespace milvus_storage
