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

#include "milvus-storage/format/lance/lance_fragment_writer.h"

#include <string>
#include <iostream>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include "lance_bridge.hpp"

namespace milvus_storage::lance {

LanceFragmentWriter::LanceFragmentWriter(std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
                                         std::shared_ptr<arrow::Schema> schema,
                                         const api::Properties& properties)
    : closed_(false), column_group_(column_group), schema_(schema), properties_(properties), dataset_(nullptr) {
  assert(column_group_);
  assert(schema_);

  // TBD: FIXME: workaround logical to get base_path
  auto file_path = column_group_->files[0].path;
  base_path_ = file_path.substr(0, file_path.find_last_of('/'));
}

class BatchIterator : public arrow::RecordBatchReader {
  public:
  BatchIterator(const std::shared_ptr<arrow::Schema>& schema,
                const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches)
      : schema_(schema), batches_(batches), position_(0) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    if (position_ >= batches_.size()) {
      *out = nullptr;
    } else {
      *out = batches_[position_++];
    }
    return arrow::Status::OK();
  }

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  size_t position_;
};

arrow::Status LanceFragmentWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  assert(!closed_);
  assert(batch->schema()->Equals(*schema_, false));
  written_rows_ += batch->num_rows();

  record_batches_.emplace_back(batch);
  return arrow::Status::OK();
}

arrow::Status LanceFragmentWriter::Flush() {
  struct ArrowArrayStream array_stream;
  assert(!closed_);

  auto batch_iterator = std::make_shared<BatchIterator>(schema_, record_batches_);
  ARROW_RETURN_NOT_OK(ExportRecordBatchReader(batch_iterator, &array_stream));

  if (!dataset_) {
    dataset_ = BlockingDataset::WriteDataset(base_path_, &array_stream);
  } else {
    dataset_->WriteArrowArrayStream(&array_stream);
  }

  record_batches_.clear();
  return arrow::Status::OK();
}

arrow::Status LanceFragmentWriter::Close() {
  assert(!closed_);
  ARROW_RETURN_NOT_OK(Flush());
  dataset_.reset();
  closed_ = true;
  return arrow::Status::OK();
}

}  // namespace milvus_storage::lance
