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

#pragma once
#include <arrow/visitor.h>
#include <arrow/result.h>
#include <memory>
#include <unordered_map>
#include <variant>

#include "milvus-storage/file/fragment.h"
#include "milvus-storage/storage/schema.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/array.h"

namespace milvus_storage {

using pk_type = std::variant<std::string_view, std::int64_t>;

// DeleteFragment is a set of deleted records
class DeleteFragment {
  public:
  DeleteFragment(arrow::fs::FileSystem& fs, std::shared_ptr<Schema> schema, int64_t id = 0);

  bool id() const { return id_; }

  void set_id(int64_t id) { id_ = id; }

  // Return true if this pk at this version have been deleted
  bool Filter(pk_type& pk, int64_t version, int64_t max_version = INT64_MAX);

  // Return true if this pk have been deleted
  bool Filter(pk_type& pk);

  arrow::Status Add(std::shared_ptr<arrow::RecordBatch> batch);
  // Make an instance of DeleteFragment of the given fragment whose type is kDelete
  static arrow::Result<DeleteFragment> Make(arrow::fs::FileSystem& fs,
                                            std::shared_ptr<Schema> schema,
                                            const Fragment& fragment);

  private:
  int64_t id_;
  std::shared_ptr<Schema> schema_;
  arrow::fs::FileSystem& fs_;
  // the deleted data parsed from the files of fragment_
  std::unordered_map<pk_type, std::vector<int64_t>> data_;  // pk to versions(if exists)
};

class DeleteFragmentVisitor : public arrow::ArrayVisitor {
  public:
  explicit DeleteFragmentVisitor(std::unordered_map<pk_type, std::vector<int64_t>> delete_set,
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
      if (delete_set_.count(value) != 0) {
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

using DeleteFragmentVector = std::vector<DeleteFragment>;

}  // namespace milvus_storage
