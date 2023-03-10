#pragma once

#include "constant_filter.h"

#include <parquet/types.h>

ConstantFilter::ConstantFilter(ComparisonType comparison_type,
                               std::string column_name, Value& value)
    : comparison_type_(comparison_type),
      Filter(std::move(column_name)),
      value_(value) {}

bool ConstantFilter::CheckStatistics(parquet::Statistics* statistics) {
  switch (statistics->physical_type()) {
    case parquet::Type::BOOLEAN:
      return CheckMinMax(dynamic_cast<parquet::BoolStatistics*>(statistics));
    case parquet::Type::INT32:
      return CheckMinMax(dynamic_cast<parquet::Int32Statistics*>(statistics));
    case parquet::Type::INT64:
      return CheckMinMax(dynamic_cast<parquet::Int64Statistics*>(statistics));
    case parquet::Type::FLOAT:
      return CheckMinMax(dynamic_cast<parquet::FloatStatistics*>(statistics));
    case parquet::Type::DOUBLE:
      return CheckMinMax(dynamic_cast<parquet::DoubleStatistics*>(statistics));
    default:
      throw StorageException("unsupported physical type");
  }
}

template <typename StatisticsType>
bool ConstantFilter::CheckMinMax(StatisticsType* statistics) {
  switch (comparison_type_) {
    case EQUAL:
      return value_ < statistics->min() && value_ > statistics->max();
    case NOT_EQUAL:
      return value_ == statistics->min() &&
             statistics->min() == statistics->max();
    case GREATER:
      return value_ >= statistics->max();
    case LESS:
      return value_ <= statistics->min();
    case GREATER_EQUAL:
      return value_ > statistics->max();
    case LESS_EQUAL:
      return value_ < statistics->min();
    default:
      throw StorageException("unsupported comparison type");
  }
}
