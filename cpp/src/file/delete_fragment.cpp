// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "file/delete_fragment.h"
#include <memory>
#include "common/status.h"
#include "common/arrow_util.h"
#include "common/macro.h"
#include "storage/options.h"
#include "arrow/array.h"
#include "reader/multi_files_sequential_reader.h"

namespace milvus_storage {
arrow::Status DeleteFragmentVisitor::Visit(const arrow::Int64Array& array) { return Visit<arrow::Int64Array>(array); }

arrow::Status DeleteFragmentVisitor::Visit(const arrow::StringArray& array) { return Visit<arrow::StringArray>(array); }

DeleteFragment::DeleteFragment(std::shared_ptr<arrow::fs::FileSystem> fs, std::shared_ptr<Schema> schema, int64_t id)
    : fs_(fs), schema_(schema), id_(id) {}

Status DeleteFragment::Add(std::shared_ptr<arrow::RecordBatch> batch) {
  auto schema_options = schema_->options();
  auto pk_col = batch->GetColumnByName(schema_options->primary_column);
  std::shared_ptr<arrow::Int64Array> version_col = nullptr;
  if (schema_->options()->has_version_column()) {
    auto tmp = batch->GetColumnByName(schema_options->version_column);
    version_col = std::static_pointer_cast<arrow::Int64Array>(tmp);
  }

  DeleteFragmentVisitor visitor(data_, version_col);
  RETURN_ARROW_NOT_OK(pk_col->Accept(&visitor));
  return Status::OK();
}

Result<DeleteFragment> DeleteFragment::Make(std::shared_ptr<arrow::fs::FileSystem> fs,
                                            std::shared_ptr<Schema> schema,
                                            const Fragment& fragment) {
  DeleteFragment delete_fragment(fs, schema, fragment.id());

  auto opts = std::make_shared<ReadOptions>();
  opts->columns = schema->delete_schema()->field_names();
  MultiFilesSequentialReader rec_reader(fs, {fragment}, schema->delete_schema(), opts);
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
