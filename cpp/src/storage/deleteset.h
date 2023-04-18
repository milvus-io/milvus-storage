#pragma once

#include <arrow/type_fwd.h>
#include "storage/default_space.h"
#include "arrow/visitor.h"
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include "arrow/array/array_primitive.h"

namespace milvus_storage {

using pk_type = std::variant<std::string_view, int64_t>;
class DeleteSet {
  public:
  explicit DeleteSet(const DefaultSpace& space);

  Status Build();

  Status Add(std::shared_ptr<arrow::RecordBatch>& batch);

  std::vector<int64_t> GetVersionByPk(pk_type& pk);

  private:
  const DefaultSpace& space_;
  std::unordered_map<pk_type, std::vector<int64_t>> data_;
  bool has_version_col_;
};

class DeleteSetVisitor : public arrow::ArrayVisitor {
  public:
  explicit DeleteSetVisitor(std::unordered_map<pk_type, std::vector<int64_t>> delete_set,
                            std::shared_ptr<arrow::Int64Array> version_col = nullptr)
      : delete_set_(std::move(delete_set)),
        version_col_(std::move(version_col)),
        has_version_col_(version_col == nullptr) {}

  arrow::Status Visit(const arrow::StringArray& array) override;

  arrow::Status Visit(const arrow::Int64Array& array) override;

  private:
  template <typename T>
  arrow::Status Visit(const T& array) {
    for (int i = 0; i < array.length(); ++i) {
      auto value = array.Value(i);
      if (!has_version_col_) {
        delete_set_.emplace(value, std::vector<int64_t>());
        continue;
      }
      if (delete_set_.contains(value)) {
        auto v = version_col_->Value(i);
        delete_set_.at(value).push_back(version_col_->Value(i));
      } else {
        delete_set_.emplace(value, std::vector<int64_t>{version_col_->Value(i)});
      }
    }
    return arrow::Status::OK();
  }

  std::unordered_map<pk_type, std::vector<int64_t>> delete_set_;
  std::shared_ptr<arrow::Int64Array> version_col_;
  bool has_version_col_;
};
}  // namespace milvus_storage