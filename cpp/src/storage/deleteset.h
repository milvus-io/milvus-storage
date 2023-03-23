#pragma once

#include <arrow/array/array_binary.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/visitor.h>

#include <memory>
#include <unordered_map>
#include <variant>

#include "storage/default_space.h"

using pk_type = std::variant<std::string_view, int64_t>;
class DeleteSet {
  public:
  explicit DeleteSet(const DefaultSpace& space);

  void
  Add(std::shared_ptr<arrow::RecordBatch>& batch);

  std::vector<int64_t>
  GetVersionByPk(pk_type& pk);

  private:
  const DefaultSpace& space_;
  std::shared_ptr<std::unordered_map<pk_type, std::vector<int64_t>>> data_;
};

class DeleteSetVisitor : public arrow::ArrayVisitor {
  public:
  DeleteSetVisitor(std::shared_ptr<std::unordered_map<pk_type, std::vector<int64_t>>>& delete_set,
                   std::shared_ptr<arrow::Int64Array>& version_col)
      : delete_set_(delete_set), version_col_(version_col) {
  }

  arrow::Status
  Visit(const arrow::StringArray& array) override;
  arrow::Status
  Visit(const arrow::Int64Array& array) override;

  private:
  std::shared_ptr<std::unordered_map<pk_type, std::vector<int64_t>>> delete_set_;
  std::shared_ptr<arrow::Int64Array> version_col_;
};
