#include "filter/value.h"

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

// TODO: add cast
bool Value::operator==(const Value& other) const { return TemplateBooleanOperation<Equal>(*this, other); }

bool Value::operator!=(const Value& other) const { return TemplateBooleanOperation<NotEqual>(*this, other); }

bool Value::operator>=(const Value& other) const {
  return TemplateBooleanOperation<GreaterThanOrEqualTo>(*this, other);
}

bool Value::operator<=(const Value& other) const { return TemplateBooleanOperation<LessThanOrEqualTo>(*this, other); }

bool Value::operator>(const Value& other) const { return TemplateBooleanOperation<GreaterThan>(*this, other); }

bool Value::operator<(const Value& other) const { return TemplateBooleanOperation<LessThan>(*this, other); }

Value Value::Bool(bool value) {
  Value v;
  v.type_ = BOOLEAN;
  v.value_.bool_value_ = value;
  return v;
}

Value Value::Int32(int32_t value) {
  Value v;
  v.type_ = INT32;
  v.value_.int32_value_ = value;
  return v;
}

Value Value::Int64(int64_t value) {
  Value v;
  v.type_ = INT64;
  v.value_.int64_value_ = value;
  return v;
}

Value Value::Float(float value) {
  Value v;
  v.type_ = FLOAT;
  v.value_.float_value_ = value;
  return v;
}

Value Value::Double(double value) {
  Value v;
  v.type_ = DOUBLE;
  v.value_.double_value_ = value;
  return v;
}

Value Value::String(std::string value) {
  Value v;
  v.type_ = STRING;
  v.string_value_ = value;
  return v;
}
}  // namespace milvus_storage
