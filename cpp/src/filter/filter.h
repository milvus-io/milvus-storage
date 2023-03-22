#pragma once

#include <bitset>
#include "constants.h"
#include "parquet/statistics.h"
using filter_mask = std::bitset<kReadBatchSize>;
class Filter {
 public:
  Filter(std::string column_name) : column_name_(column_name) {}
  virtual bool CheckStatistics(parquet::Statistics *) = 0;
  std::string get_column_name() { return column_name_; };
  virtual void Apply(arrow::Array *col_data, filter_mask &bitset) = 0;

 protected:
  std::string column_name_;
};