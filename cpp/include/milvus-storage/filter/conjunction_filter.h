

#pragma once
#include "milvus-storage/filter/filter.h"

#include <utility>

namespace milvus_storage {

class ConjunctionOrFilter : public Filter {
  public:
  explicit ConjunctionOrFilter(const FilterSet& filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(filters) {}

  bool CheckStatistics(parquet::Statistics* stats) override;

  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  const FilterSet& filters_;
};

class ConjunctionAndFilter : public Filter {
  public:
  explicit ConjunctionAndFilter(const FilterSet& filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(filters) {}

  bool CheckStatistics(parquet::Statistics* stats) override;

  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  const FilterSet& filters_;
};
}  // namespace milvus_storage
