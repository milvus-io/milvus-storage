#include "storage/deleteset.h"
#include <arrow/type_fwd.h>
#include "arrow/array/array_primitive.h"
#include "reader/scan_record_reader.h"
#include "arrow/array/array_binary.h"
namespace milvus_storage {

arrow::Status DeleteSetVisitor::Visit(const arrow::Int64Array& array) { return Visit<arrow::Int64Array>(array); }

arrow::Status DeleteSetVisitor::Visit(const arrow::StringArray& array) { return Visit<arrow::StringArray>(array); }

DeleteSet::DeleteSet(const DefaultSpace& space) : space_(space) {
  has_version_col_ = space_.schema_->options()->has_version_column();
}

Status DeleteSet::Build() {
  const auto& delete_files = space_.manifest_->delete_files();
  auto option = std::make_shared<ReadOptions>();
  ScanRecordReader rec_reader(option, delete_files, space_);

  for (const auto& batch_rec : rec_reader) {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto batch, batch_rec);
    Add(batch);
  }
  RETURN_ARROW_NOT_OK(rec_reader.Close());
  return Status::OK();
}

Status DeleteSet::Add(std::shared_ptr<arrow::RecordBatch>& batch) {
  auto schema_options = space_.schema_->options();
  auto pk_col = batch->GetColumnByName(schema_options->primary_column);
  std::shared_ptr<arrow::Int64Array> version_col = nullptr;
  if (has_version_col_) {
    auto tmp = batch->GetColumnByName(schema_options->version_column);
    version_col = std::static_pointer_cast<arrow::Int64Array>(tmp);
  }

  DeleteSetVisitor visitor(data_, version_col);
  RETURN_ARROW_NOT_OK(pk_col->Accept(&visitor));
  return Status::OK();
}

std::vector<int64_t> DeleteSet::GetVersionByPk(pk_type& pk) {
  if (data_.contains(pk)) {
    return data_.at(pk);
  }
  return {};
}
}  // namespace milvus_storage