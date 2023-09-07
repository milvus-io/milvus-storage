#pragma once
#include "filter/filter.h"

#include <utility>

namespace milvus_storage {

class ConjunctionOrFilter : public Filter {
  public:
  explicit ConjunctionOrFilter(std::vector<std::unique_ptr<Filter>>& filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(filters) {}

  bool CheckStatistics(parquet::Statistics* stats) override;

  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  std::vector<std::unique_ptr<Filter>>& filters_;
};

class ConjunctionAndFilter : public Filter {
  public:
  explicit ConjunctionAndFilter(std::vector<std::unique_ptr<Filter>>& filters, std::string column_name)
      : Filter(std::move(column_name)), filters_(filters) {}

  bool CheckStatistics(parquet::Statistics* stats) override;

  Status Apply(arrow::Array* col_data, filter_mask& bitset) override;

  private:
  std::vector<std::unique_ptr<Filter>>& filters_;
};
}  // namespace milvus_storage
