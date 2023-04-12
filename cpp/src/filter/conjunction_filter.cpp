#include "conjunction_filter.h"
#include "common/macro.h"

namespace milvus_storage {

bool ConjunctionOrFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (!filter.CheckStatistics(stats)) {
      return false;
    }
  }
  return true;
}

Status ConjunctionOrFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  filter_mask or_bitset;
  for (auto& filter : filters_) {
    filter_mask bitset_cloned = bitset;
    RETURN_NOT_OK(filter.Apply(col_data, bitset_cloned));
    or_bitset &= bitset_cloned;
  }
  bitset |= or_bitset;
  return Status::OK();
}

bool ConjunctionAndFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (filter.CheckStatistics(stats)) {
      return true;
    }
  }
  return false;
}

Status ConjunctionAndFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  for (auto& filter : filters_) {
    RETURN_NOT_OK(filter.Apply(col_data, bitset));
  }
  return Status::OK();
}
}  // namespace milvus_storage