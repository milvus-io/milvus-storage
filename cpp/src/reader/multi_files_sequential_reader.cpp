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

#include "reader/multi_files_sequential_reader.h"
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include "common/arrow_util.h"

namespace milvus_storage {

MultiFilesSequentialReader::MultiFilesSequentialReader(arrow::fs::FileSystem& fs,
                                                       const FragmentVector& fragments,
                                                       std::shared_ptr<arrow::Schema> schema,
                                                       const ReadOptions& options)
    : fs_(fs), schema_(std::move(schema)), options_(options) {
  for (const auto& fragment : fragments) {
    files_.insert(files_.end(), fragment.files().begin(), fragment.files().end());
  }
}

std::shared_ptr<arrow::Schema> MultiFilesSequentialReader::schema() const { return schema_; }

arrow::Status MultiFilesSequentialReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  while (true) {
    if (!curr_reader_) {
      if (next_pos_ >= files_.size()) {
        batch = nullptr;
        return arrow::Status::OK();
      }

      auto s = MakeArrowFileReader(fs_, files_[next_pos_++]);
      if (!s.ok()) {
        return arrow::Status::UnknownError(s.status().ToString());
      }
      holding_file_reader_ = s.value();

      auto s2 = MakeArrowRecordBatchReader(holding_file_reader_, options_);
      if (!s2.ok()) {
        return arrow::Status::UnknownError(s2.status().ToString());
      }
      curr_reader_ = s2.value();
    }

    std::shared_ptr<arrow::RecordBatch> tmp_batch;
    auto s = curr_reader_->ReadNext(&tmp_batch);
    if (!s.ok()) {
      return s;
    }

    if (tmp_batch == nullptr) {
      curr_reader_ = nullptr;
      holding_file_reader_ = nullptr;
      continue;
    }

    *batch = tmp_batch;
    return arrow::Status::OK();
  }
}

}  // namespace milvus_storage
