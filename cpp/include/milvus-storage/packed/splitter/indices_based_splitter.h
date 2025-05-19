
#pragma once

#include <vector>
#include <map>
#include "milvus-storage/packed/splitter/splitter_plugin.h"

namespace milvus_storage {

class IndicesBasedSplitter : public SplitterPlugin {
  public:
  explicit IndicesBasedSplitter(const std::vector<std::vector<int>>& column_indices);

  void Init() override;

  std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) override;

  private:
  std::vector<std::vector<int>> column_indices_;
};

}  // namespace milvus_storage
