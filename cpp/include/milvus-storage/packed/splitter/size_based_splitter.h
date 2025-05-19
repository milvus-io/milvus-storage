
#pragma once

#include "milvus-storage/packed/splitter/splitter_plugin.h"
#include "milvus-storage/common/result.h"

namespace milvus_storage {

class SizeBasedSplitter : public SplitterPlugin {
  public:
  /*
   * @brief SizeBasedSplitter is a splitter plugin that splits record batches into column groups based on the size of
   * each column.
   */
  explicit SizeBasedSplitter(size_t max_group_size);

  void Init() override;

  std::vector<ColumnGroup> SplitRecordBatches(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches);

  std::vector<ColumnGroup> Split(const std::shared_ptr<arrow::RecordBatch>& record) override;

  size_t max_group_size_;
  static constexpr size_t SPLIT_THRESHOLD = 1024;  // 1K
};

}  // namespace milvus_storage
