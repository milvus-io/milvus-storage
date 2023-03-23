#pragma once
#include "common/exception.h"
#include "filter/filter.h"

#include <utility>
class ConjunctionOrFilter : public Filter {
  public:
  explicit ConjunctionOrFilter(std::vector<Filter> filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(std::move(filters)) {
    if (filters.empty()) {
      throw StorageException("ConjunctionOrFilter must have at least one filter");
    }
  }
  bool
  CheckStatistics(parquet::Statistics* stats) override;
  void
  Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  std::vector<Filter> filters_;
};

class ConjunctionAndFilter : public Filter {
  public:
  explicit ConjunctionAndFilter(std::vector<Filter> filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(std::move(filters)) {
    if (filters.empty()) {
      throw StorageException("ConjunctionAndFilter must have at least one filter");
    }
  }
  bool
  CheckStatistics(parquet::Statistics* stats) override;
  void
  Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  std::vector<Filter> filters_;
};
