#pragma once

#include "storage/default_space.h"
#include "arrow/visitor.h"
#include <variant>
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
};

class DeleteSetVisitor : public arrow::ArrayVisitor {
  public:
  DeleteSetVisitor(std::unordered_map<pk_type, std::vector<int64_t>>& delete_set,
                   std::shared_ptr<arrow::Int64Array>& version_col)
      : delete_set_(delete_set), version_col_(version_col) {}

  arrow::Status Visit(const arrow::StringArray& array) override;

  arrow::Status Visit(const arrow::Int64Array& array) override;

  private:
  std::unordered_map<pk_type, std::vector<int64_t>>& delete_set_;
  std::shared_ptr<arrow::Int64Array> version_col_;
};
}  // namespace milvus_storage