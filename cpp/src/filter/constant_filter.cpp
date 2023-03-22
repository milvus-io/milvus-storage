#include "constant_filter.h"

#include <arrow/array/array_primitive.h>
#include <arrow/type_fwd.h>
#include <parquet/types.h>
#include "exception.h"

ConstantFilter::ConstantFilter(ComparisonType comparison_type, std::string column_name, Value &value)
    : comparison_type_(comparison_type), Filter(std::move(column_name)), value_(value) {}

bool ConstantFilter::CheckStatistics(parquet::Statistics *statistics) {
  switch (statistics->physical_type()) {
    case parquet::Type::BOOLEAN:
      return CheckMinMax(dynamic_cast<parquet::BoolStatistics *>(statistics));
    case parquet::Type::INT32:
      return CheckMinMax(dynamic_cast<parquet::Int32Statistics *>(statistics));
    case parquet::Type::INT64:
      return CheckMinMax(dynamic_cast<parquet::Int64Statistics *>(statistics));
    case parquet::Type::FLOAT:
      return CheckMinMax(dynamic_cast<parquet::FloatStatistics *>(statistics));
    case parquet::Type::DOUBLE:
      return CheckMinMax(dynamic_cast<parquet::DoubleStatistics *>(statistics));
    default:
      throw StorageException("unsupported physical type");
  }
}

template <typename StatisticsType>
bool ConstantFilter::CheckMinMax(StatisticsType *statistics) {
  switch (comparison_type_) {
    case EQUAL:
      return value_ < statistics->min() && value_ > statistics->max();
    case NOT_EQUAL:
      return value_ == statistics->min() && statistics->min() == statistics->max();
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

void ConstantFilter::Apply(arrow::Array *col_data, filter_mask &bitset) {
  switch (col_data->type_id()) {
    case arrow::Type::BOOL:
      ApplyFilter<arrow::BooleanArray>(dynamic_cast<arrow::BooleanArray *>(col_data), bitset);
      break;
    case arrow::Type::INT8:
      ApplyFilter<arrow::Int8Array>(dynamic_cast<arrow::Int8Array *>(col_data), bitset);
      break;
    case arrow::Type::UINT8:
      ApplyFilter<arrow::UInt8Array>(dynamic_cast<arrow::UInt8Array *>(col_data), bitset);
      break;
    case arrow::Type::INT16:
      ApplyFilter<arrow::Int16Array>(dynamic_cast<arrow::Int16Array *>(col_data), bitset);
      break;
    case arrow::Type::UINT16:
      ApplyFilter<arrow::UInt16Array>(dynamic_cast<arrow::UInt16Array *>(col_data), bitset);
      break;
    case arrow::Type::INT32:
      ApplyFilter<arrow::Int32Array>(dynamic_cast<arrow::Int32Array *>(col_data), bitset);
      break;
    case arrow::Type::UINT32:
      ApplyFilter<arrow::UInt32Array>(dynamic_cast<arrow::UInt32Array *>(col_data), bitset);
      break;
    case arrow::Type::INT64:
      ApplyFilter<arrow::Int64Array>(dynamic_cast<arrow::Int64Array *>(col_data), bitset);
      break;
    case arrow::Type::UINT64:
      ApplyFilter<arrow::UInt64Array>(dynamic_cast<arrow::UInt64Array *>(col_data), bitset);
      break;
    case arrow::Type::FLOAT:
      ApplyFilter<arrow::FloatArray>(dynamic_cast<arrow::FloatArray *>(col_data), bitset);
      break;
    case arrow::Type::DOUBLE:
      ApplyFilter(dynamic_cast<arrow::DoubleArray *>(col_data), bitset);
      break;
    default:
      throw StorageException("unsupported type");
  }
}

template <typename T>
bool checkValue(T value, T target, ComparisonType comparison_type) {
  switch (comparison_type) {
    case EQUAL:
      return value != target;
    case NOT_EQUAL:
      return value == target;
    case LESS:
      return value <= target;
    case LESS_EQUAL:
      return value < target;
    case GREATER:
      return value >= target;
    case GREATER_EQUAL:
      return value > target;
    default:
      throw StorageException("unsupported comparison type");
  }
}

template <typename ArrayType, typename T>
void ConstantFilter::ApplyFilter(const ArrayType *array, filter_mask &bitset) {
  for (int i = 0; i < array->length(); i++) {
    if (checkValue(value_.get_value<typename T::c_type>(), array->Value(i), comparison_type_)) {
      bitset.set(i);
    }
  }
}
