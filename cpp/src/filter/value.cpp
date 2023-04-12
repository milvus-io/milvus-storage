#include "value.h"

#include <cstdint>

namespace milvus_storage {

template <>
bool Value::get_value() const {
  return value_.bool_value_;
}

template <>
int8_t Value::get_value() const {
  return value_.int8_value_;
}

template <>
int16_t Value::get_value() const {
  return value_.int16_value_;
}

template <>
int32_t Value::get_value() const {
  return value_.int32_value_;
}

template <>
int64_t Value::get_value() const {
  return value_.int64_value_;
}

template <>
float Value::get_value() const {
  return value_.float_value_;
}

template <>
double Value::get_value() const {
  return value_.double_value_;
}

template <>
std::string Value::get_value() const {
  return string_value_;
}

bool Value::operator==(const Value& other) const { return TemplateBooleanOperation<Equal>(*this, other); }

bool Value::operator!=(const Value& other) const { return TemplateBooleanOperation<NotEqual>(*this, other); }

bool Value::operator>=(const Value& other) const {
  return TemplateBooleanOperation<GreaterThanOrEqualTo>(*this, other);
}

bool Value::operator<=(const Value& other) const { return TemplateBooleanOperation<LessThanOrEqualTo>(*this, other); }

bool Value::operator>(const Value& other) const { return TemplateBooleanOperation<GreaterThan>(*this, other); }

bool Value::operator<(const Value& other) const { return TemplateBooleanOperation<LessThan>(*this, other); }
}  // namespace milvus_storage