#pragma once

#include "parquet/statistics.h"
class Filter {
 public:
  Filter(std::string column_name) : column_name_(column_name) {}
  virtual bool CheckStatistics(parquet::Statistics*) = 0;
  std::string get_column_name() { return column_name_; };

 protected:
  std::string column_name_;
};