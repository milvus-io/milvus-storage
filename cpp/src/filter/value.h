#pragma once

#include <cstdint>
#include <string>

namespace milvus_storage {

enum LogicType {
  BOOLEAN,
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  STRING,
};
class Value {
  public:
  Value() = default;
  Value(int32_t value) : type_(INT32) { value_.int32_value_ = value; }   // NOLINT
  Value(int64_t value) : type_(INT64) { value_.int64_value_ = value; }   // NOLINT
  Value(bool value) : type_(BOOLEAN) { value_.bool_value_ = value; }     // NOLINT
  Value(float value) : type_(FLOAT) { value_.float_value_ = value; }     // NOLINT
  Value(double value) : type_(DOUBLE) { value_.double_value_ = value; }  // NOLINT
  Value(std::string value) : type_(STRING) { string_value_ = value; }    // NOLINT
  explicit Value(LogicType type) : type_(type) {}                        // NOLINT

  LogicType get_logic_type() const { return type_; }

  template <typename T>
  T get_value() const {}

  bool operator==(const Value& other) const;
  bool operator!=(const Value& other) const;
  bool operator>=(const Value& other) const;
  bool operator<=(const Value& other) const;
  bool operator>(const Value& other) const;
  bool operator<(const Value& other) const;

  static Value Bool(bool value);
  static Value Int32(int32_t value);
  static Value Int64(int64_t value);
  static Value Float(float value);
  static Value Double(double value);
  static Value String(std::string value);

  private:
  union Val {
    bool bool_value_;
    int8_t int8_value_;
    int16_t int16_value_;
    int32_t int32_value_;
    int64_t int64_value_;
    float float_value_;
    double double_value_;
  } value_;

  std::string string_value_;

  LogicType type_;
};

template <>
bool Value::get_value() const;

template <>
int8_t Value::get_value() const;

template <>
int16_t Value::get_value() const;

template <>
int32_t Value::get_value() const;

template <>
int64_t Value::get_value() const;

template <>
float Value::get_value() const;

template <>
double Value::get_value() const;

template <>
std::string Value::get_value() const;

struct Equal {
  template <typename T>
  static bool Operation(T a, T b) {
    return a == b;
  }
};

struct NotEqual {
  template <typename T>
  static bool Operation(T a, T b) {
    return !Equal::Operation(a, b);
  }
};

struct GreaterThan {
  template <typename T>
  static bool Operation(T a, T b) {
    return a > b;
  }
};

struct LessThan {
  template <typename T>
  static bool Operation(T a, T b) {
    return a < b;
  }
};

struct GreaterThanOrEqualTo {
  template <typename T>
  static bool Operation(T a, T b) {
    return a >= b;
  }
};

struct LessThanOrEqualTo {
  template <typename T>
  static bool Operation(T a, T b) {
    return a <= b;
  }
};

template <typename OpType>
static bool TemplateBooleanOperation(const Value& a, const Value& b) {
  LogicType left = a.get_logic_type(), right = b.get_logic_type();

  switch (left) {
    case BOOLEAN:
      return OpType::Operation(a.get_value<bool>(), b.get_value<bool>());
    case INT8:
      return OpType::Operation(static_cast<int32_t>(a.get_value<int8_t>()), b.get_value<int32_t>());
    case INT16:
      return OpType::Operation(static_cast<int32_t>(a.get_value<int16_t>()), b.get_value<int32_t>());
    case INT32:
      return OpType::Operation(a.get_value<int32_t>(), b.get_value<int32_t>());
    case INT64:
      return OpType::Operation(a.get_value<int64_t>(), b.get_value<int64_t>());
    case FLOAT:
      return OpType::Operation(a.get_value<float>(), b.get_value<float>());
    case DOUBLE:
      return OpType::Operation(a.get_value<double>(), b.get_value<double>());
    case STRING:
      return OpType::Operation(a.get_value<std::string>(), b.get_value<std::string>());
  }
  return false;
}

}  // namespace milvus_storage
