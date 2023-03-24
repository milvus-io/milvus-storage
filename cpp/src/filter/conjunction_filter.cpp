#include "conjunction_filter.h"

bool
ConjunctionOrFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (!filter.CheckStatistics(stats)) {
      return false;
    }
  }
  return true;
}

void
ConjunctionOrFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  filter_mask or_bitset;
  for (auto& filter : filters_) {
    filter_mask bitset_cloned = bitset;
    filter.Apply(col_data, bitset_cloned);
    or_bitset &= bitset_cloned;
  }
  bitset |= or_bitset;
}

bool
ConjunctionAndFilter::CheckStatistics(parquet::Statistics* stats) {
  for (auto& filter : filters_) {
    if (filter.CheckStatistics(stats)) {
      return true;
    }
  }
  return false;
}

void
ConjunctionAndFilter::Apply(arrow::Array* col_data, filter_mask& bitset) {
  for (auto& filter : filters_) {
    filter.Apply(col_data, bitset);
  }
}
