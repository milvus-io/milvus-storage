#pragma once
#include "arrow/array/array_base.h"
#include "filter.h"
#include "value.h"

namespace milvus_storage {

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
  ConstantFilter(ComparisonType comparison_type, std::string column_name, Value& value);

  bool CheckStatistics(parquet::Statistics*) override;
  
  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  template <typename StatisticsType>
  bool CheckMinMax(StatisticsType* statistics);

  template <typename ArrayType, typename T = typename ArrayType::TypeClass>
  void ApplyFilter(const ArrayType* array, filter_mask& bitset);

  ComparisonType comparison_type_;
  Value value_;
};
}  // namespace milvus_storage