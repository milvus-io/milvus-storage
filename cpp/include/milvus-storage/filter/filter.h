

#pragma once

#include <bitset>
#include <memory>
#include "arrow/record_batch.h"
#include "milvus-storage/common/constants.h"
#include "parquet/statistics.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/common/macro.h"

namespace milvus_storage {

using filter_mask = std::bitset<kReadBatchSize>;
class Filter {
  public:
  using FilterSet = std::vector<Filter*>;
  explicit Filter(std::string column_name) : column_name_(std::move(column_name)) {}

  virtual bool CheckStatistics(parquet::Statistics*) = 0;

  std::string get_column_name() { return column_name_; };

  virtual Status Apply(arrow::Array* col_data, filter_mask& bitset) = 0;

  static Status ApplyFilter(const std::shared_ptr<arrow::RecordBatch>& record_batch,
                            const FilterSet& filters,
                            filter_mask& bitset) {
    for (auto& filter : filters) {
      auto col_data = record_batch->GetColumnByName(filter->get_column_name());
      RETURN_NOT_OK(filter->Apply(col_data.get(), bitset));
    }
    return Status::OK();
  }

  virtual ~Filter(){};

  protected:
  std::string column_name_;
};
}  // namespace milvus_storage
