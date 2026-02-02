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

#ifdef BUILD_GTEST

#include "milvus-storage/format/lance/lance_table_writer.h"

#include <string>
#include <iostream>
#include <unordered_set>

#include <arrow/chunked_array.h>  // keep this line before other arrow header
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/status.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include "lance_bridge.h"

namespace milvus_storage::lance {

LanceTableWriter::LanceTableWriter(const std::string& base_path,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const api::Properties& properties)
    : closed_(false), base_path_(base_path), schema_(schema), properties_(properties), dataset_(nullptr) {
  assert(schema_);
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

arrow::Status LanceTableWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  assert(!closed_);
  assert(batch->schema()->Equals(*schema_, false));
  written_rows_ += batch->num_rows();

  record_batches_.emplace_back(batch);
  return arrow::Status::OK();
}

arrow::Status LanceTableWriter::Flush() { return arrow::Status::OK(); }

bool fids_contains(const std::vector<uint64_t>& origin, const std::vector<uint64_t>& current) {
  assert(current.size() > origin.size());
  std::unordered_set<uint64_t> current_set(current.begin(), current.end());
  for (uint64_t elem : origin) {
    if (current_set.find(elem) == current_set.end()) {
      return false;
    }
  }
  return true;
}

std::vector<uint64_t> fids_diff(const std::vector<uint64_t>& origin, const std::vector<uint64_t>& current) {
  assert(current.size() > origin.size());

  std::unordered_set<uint64_t> origin_set(origin.begin(), origin.end());
  std::vector<uint64_t> diff;

  for (uint64_t elem : current) {
    if (origin_set.find(elem) == origin_set.end()) {
      diff.emplace_back(elem);
    }
  }
  return diff;
}

arrow::Result<api::ColumnGroupFile> LanceTableWriter::Close() {
  assert(!closed_);
  struct ArrowArrayStream array_stream;

  auto batch_iterator = std::make_shared<BatchIterator>(schema_, record_batches_);
  ARROW_RETURN_NOT_OK(ExportRecordBatchReader(batch_iterator, &array_stream));

  if (!dataset_) {
    try {
      dataset_ = BlockingDataset::OpenUnique(base_path_);
      origin_fids_ = dataset_->GetAllFragmentIds();
      dataset_->WriteArrowArrayStream(&array_stream);
    } catch (std::exception& e) {
      // dataset no exist
      origin_fids_.clear();
      dataset_ = BlockingDataset::WriteDataset(base_path_, &array_stream);
    }
  } else {
    dataset_->WriteArrowArrayStream(&array_stream);
  }
  record_batches_.clear();

  std::vector<uint64_t> append_fids;
  std::vector<uint64_t> current_fids;
  current_fids = dataset_->GetAllFragmentIds();

  if (current_fids.size() < origin_fids_.size()) {
    return arrow::Status::Invalid(
        fmt::format("LanceTableWriter: current fragment ids size is less than origin fragment ids size [current "
                    "size={}, origin size={}]",
                    current_fids.size(),  // NOLINT
                    origin_fids_.size()));
  }

  if (current_fids.size() == origin_fids_.size()) {
    return api::ColumnGroupFile{.path = base_path_, .start_index = 0, .end_index = written_rows_};
  }

  if (!fids_contains(origin_fids_, current_fids)) {
    return arrow::Status::Invalid("LanceTableWriter: current fragment ids is not a superset of origin fragment ids");
  }

  append_fids = fids_diff(origin_fids_, current_fids);
  assert(append_fids.size() == 1);

  dataset_.reset();
  closed_ = true;
  return api::ColumnGroupFile{
      .path = base_path_ + "?fragment_id=" + std::to_string(append_fids[0]),
      .start_index = 0,
      .end_index = written_rows_,
  };
}

}  // namespace milvus_storage::lance

#endif  // BUILD_GTEST
