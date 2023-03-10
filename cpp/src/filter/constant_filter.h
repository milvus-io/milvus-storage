#pragma once
#include "filter.h"
#include "value.h"

enum ComparisonType {
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
};

class ConstantFilter : public Filter {
 public:
  ConstantFilter(ComparisonType comparison_type, std::string column_name,
                 Value &value);
  bool CheckStatistics(parquet::Statistics *) override;
  template <typename StatisticsType>
  bool CheckMinMax(StatisticsType *statistics);

 private:
  ComparisonType comparison_type_;
  Value value_;
};