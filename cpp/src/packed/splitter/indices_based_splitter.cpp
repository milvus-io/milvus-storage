

#include "milvus-storage/packed/splitter/indices_based_splitter.h"
#include "milvus-storage/packed/column_group.h"

namespace milvus_storage {

IndicesBasedSplitter::IndicesBasedSplitter(const std::vector<std::vector<int>>& column_indices)
    : column_indices_(column_indices) {}

void IndicesBasedSplitter::Init() {}

std::vector<ColumnGroup> IndicesBasedSplitter::Split(const std::shared_ptr<arrow::RecordBatch>& record) {
  std::vector<ColumnGroup> column_groups;

  for (GroupId group_id = 0; group_id < column_indices_.size(); group_id++) {
    auto batch = record->SelectColumns(column_indices_[group_id]).ValueOrDie();
    column_groups.push_back(ColumnGroup(group_id, column_indices_[group_id], batch));
  }

  return column_groups;
}

}  // namespace milvus_storage
