#include "storage/deleteset.h"
#include "arrow/array/array_primitive.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "common/exception.h"
#include "common/macro.h"
#include "parquet/exception.h"
#include "reader/scan_record_reader.h"
#include "storage/default_space.h"
#include "storage/manifest.h"
#include <arrow/array/array_binary.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>
#include <cstdint>
#include <memory>

arrow::Status
DeleteSetVisitor::Visit(const arrow::Int64Array& array) {
  for (int i = 0; i < array.length(); ++i) {
    auto value = array.Value(i);
    if (delete_set_->contains(value)) {
      delete_set_->at(value).push_back(version_col_->Value(i));
    } else {
      delete_set_->emplace(value, std::vector<int64_t>{version_col_->Value(i)});
    }
  }
  return arrow::Status::OK();
}

arrow::Status
DeleteSetVisitor::Visit(const arrow::StringArray& array) {
  for (int i = 0; i < array.length(); ++i) {
    auto value = array.Value(i);
    if (delete_set_->contains(value)) {
      delete_set_->at(value).push_back(version_col_->Value(i));
    } else {
      delete_set_->emplace(value, std::vector<int64_t>());
    }
  }
  return arrow::Status::OK();
}

DeleteSet::DeleteSet(const DefaultSpace& space) : space_(space) {
  data_ = std::make_shared<std::unordered_map<pk_type, std::vector<int64_t>>>();
  auto pk_type = space_.manifest_->get_schema()->GetFieldByName(space_.options_->primary_column)->type();
  if (pk_type->id() != arrow::Type::INT64 && pk_type->id() != arrow::Type::STRING) {
    throw StorageException("primary key type must be string or int64");
  }

  auto version_type = space.manifest_->get_schema()->GetFieldByName(space.options_->version_column)->type();
  if (version_type->id() != arrow::Type::INT64) {
    throw StorageException("version type must be int64");
  }

  const auto& delete_files = space.manifest_->GetDeleteFiles();
  auto option = std::make_shared<ReadOption>();
  ScanRecordReader rec_reader(option, delete_files, space_);

  for (const auto& batch_rec : rec_reader) {
    ASSIGN_OR_RETURN_NOT_OK(auto batch, batch_rec);
    Add(batch);
    RETURN_IGNORE_NOT_OK(rec_reader.Close());
  }
}

void
DeleteSet::Add(std::shared_ptr<arrow::RecordBatch>& batch) {
  auto pk_col = batch->GetColumnByName(space_.options_->primary_column);
  auto vec_col = batch->GetColumnByName(space_.options_->version_column);

  auto int64_version_col = std::static_pointer_cast<arrow::Int64Array>(vec_col);
  DeleteSetVisitor visitor(data_, int64_version_col);
  PARQUET_THROW_NOT_OK(pk_col->Accept(&visitor));
}

std::vector<int64_t>
DeleteSet::GetVersionByPk(pk_type& pk) {
  if (data_->contains(pk)) {
    return data_->at(pk);
  }
  return {};
}
