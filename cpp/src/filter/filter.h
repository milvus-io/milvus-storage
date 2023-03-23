#pragma once

#include <arrow/type_fwd.h>
#include <bitset>
#include <memory>
#include "arrow/record_batch.h"
#include "common/constants.h"
#include "parquet/statistics.h"
using filter_mask = std::bitset<kReadBatchSize>;
class Filter {
 public:
  Filter(std::string column_name) : column_name_(column_name) {}
  virtual bool CheckStatistics(parquet::Statistics *) = 0;
  std::string get_column_name() { return column_name_; };
  virtual void Apply(arrow::Array *col_data, filter_mask &bitset) = 0;
  static void ApplyFilter(const std::shared_ptr<arrow::RecordBatch> &record_batch, std::vector<Filter *> &filters,
                          filter_mask &bitset) {
    for (auto &filter : filters) {
      auto col_data = record_batch->GetColumnByName(filter->get_column_name());
      filter->Apply(col_data.get(), bitset);
    }
  }

 protected:
  std::string column_name_;
};
