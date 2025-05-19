

#include "milvus-storage/file/delete_fragment.h"
#include <memory>
#include "milvus-storage/common/status.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/storage/options.h"
#include "arrow/array.h"
#include "milvus-storage/reader/multi_files_sequential_reader.h"

namespace milvus_storage {
arrow::Status DeleteFragmentVisitor::Visit(const arrow::Int64Array& array) { return Visit<arrow::Int64Array>(array); }

arrow::Status DeleteFragmentVisitor::Visit(const arrow::StringArray& array) { return Visit<arrow::StringArray>(array); }

DeleteFragment::DeleteFragment(arrow::fs::FileSystem& fs, std::shared_ptr<Schema> schema, int64_t id)
    : fs_(fs), schema_(schema), id_(id) {}

Status DeleteFragment::Add(std::shared_ptr<arrow::RecordBatch> batch) {
  auto schema_options = schema_->options();
  auto pk_col = batch->GetColumnByName(schema_options.primary_column);
  std::shared_ptr<arrow::Int64Array> version_col = nullptr;
  if (schema_->options().has_version_column()) {
    auto tmp = batch->GetColumnByName(schema_options.version_column);
    version_col = std::static_pointer_cast<arrow::Int64Array>(tmp);
  }

  DeleteFragmentVisitor visitor(data_, version_col);
  RETURN_ARROW_NOT_OK(pk_col->Accept(&visitor));
  return Status::OK();
}

Result<DeleteFragment> DeleteFragment::Make(arrow::fs::FileSystem& fs,
                                            std::shared_ptr<Schema> schema,
                                            const Fragment& fragment) {
  DeleteFragment delete_fragment(fs, schema, fragment.id());

  MultiFilesSequentialReader rec_reader(fs, {fragment}, schema->delete_schema(), schema->options(), {});
  for (const auto& batch_rec : rec_reader) {
    ASSIGN_OR_RETURN_ARROW_NOT_OK(auto batch, batch_rec);
    delete_fragment.Add(batch);
  }
  RETURN_ARROW_NOT_OK(rec_reader.Close());
  return delete_fragment;
}

bool DeleteFragment::Filter(pk_type& pk, int64_t version, int64_t max_version) {
  if (data_.find(pk) == data_.end()) {
    return false;
  }
  std::vector<int64_t> versions = data_.at(pk);
  for (auto i : versions) {
    if (i >= version && i <= max_version) {
      return true;
    }
  }
  return false;
}

bool DeleteFragment::Filter(pk_type& pk) { return data_.find(pk) != data_.end(); }
}  // namespace milvus_storage
