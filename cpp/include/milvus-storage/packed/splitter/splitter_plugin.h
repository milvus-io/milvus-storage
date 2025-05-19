

#pragma once

#include <vector>
#include <memory>
#include <arrow/record_batch.h>
#include <milvus-storage/packed/column_group.h>

using namespace arrow;

namespace milvus_storage {

class SplitterPlugin {
  public:
  virtual ~SplitterPlugin() = default;

  virtual void Init() = 0;

  // Split the input record batch into multiple groups of columns
  virtual std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) = 0;
};

}  // namespace milvus_storage
