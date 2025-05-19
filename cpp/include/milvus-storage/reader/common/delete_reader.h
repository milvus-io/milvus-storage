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

#include <memory>
#include <utility>
#include "arrow/record_batch.h"
#include "arrow/array/array_primitive.h"
#include "milvus-storage/file/delete_fragment.h"
#include "milvus-storage/storage/options.h"
#include "arrow/visitor.h"

namespace milvus_storage {

// DeleteMergeReader filters the deleted record.
class DeleteMergeReader : public arrow::RecordBatchReader {
  public:
  class RecordBatchWithDeltedOffsets;
  class DeleteFilterVisitor;

  static std::unique_ptr<DeleteMergeReader> Make(std::unique_ptr<arrow::RecordBatchReader> reader,
                                                 const SchemaOptions& schema_options,
                                                 const DeleteFragmentVector& delete_fragments,
                                                 const ReadOptions& options);
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  DeleteMergeReader(std::unique_ptr<arrow::RecordBatchReader> reader,
                    DeleteFragmentVector delete_fragments,
                    const SchemaOptions& schema_options,
                    const ReadOptions& options)
      : reader_(std::move(reader)),
        delete_fragments_(std::move(delete_fragments)),
        schema_options_(schema_options),
        options_(options) {}

  private:
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<RecordBatchWithDeltedOffsets> filtered_batch_reader_;
  DeleteFragmentVector delete_fragments_;
  const SchemaOptions schema_options_;
  const ReadOptions options_;
};

// RecordBatchWithDeltedOffsets is reader helper to fetch records not deleted without copy
class DeleteMergeReader::RecordBatchWithDeltedOffsets {
  public:
  RecordBatchWithDeltedOffsets(std::shared_ptr<arrow::RecordBatch> batch, std::vector<int> deleted_offsets)
      : batch_(std::move(batch)), deleted_offsets_(std::move(deleted_offsets)) {}

  std::shared_ptr<arrow::RecordBatch> Next();

  private:
  std::shared_ptr<arrow::RecordBatch> batch_;
  std::vector<int> deleted_offsets_;
  int next_pos_ = 0;
  int start_offset_ = 0;
};

class DeleteMergeReader::DeleteFilterVisitor : public arrow::ArrayVisitor {
  public:
  explicit DeleteFilterVisitor(DeleteFragmentVector delete_fragments,
                               std::shared_ptr<arrow::Int64Array> version_col = nullptr,
                               int64_t version = -1)
      : version_col_(std::move(version_col)), delete_fragments_(std::move(delete_fragments)), version_(version){};

  arrow::Status Visit(const arrow::Int64Array& array) override;
  arrow::Status Visit(const arrow::StringArray& array) override;

  std::vector<int> GetOffsets() { return offsets_; }

  private:
  template <typename T>
  arrow::Status VisitTemplate(const T& array) {
    for (int i = 0; i < array.length(); i++) {
      pk_type pk = array.Value(i);
      for (auto& delete_fragment : delete_fragments_) {
        if (version_col_ != nullptr) {
          if (delete_fragment.Filter(pk, version_col_->Value(i), version_)) {
            offsets_.push_back(i);
            break;
          }
        } else {
          if (delete_fragment.Filter(pk)) {
            offsets_.push_back(i);
            break;
          }
        }
      }
    }

    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::Int64Array> version_col_;
  DeleteFragmentVector delete_fragments_;
  std::vector<int> offsets_;
  int64_t version_;
};
}  // namespace milvus_storage
