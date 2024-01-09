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

#include <arrow/type_fwd.h>
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"
#include "common/result.h"
#include <utility>
#include "storage/options.h"

namespace milvus_storage {

// FilterReader filters data by the filters passed by read options.
class FilterReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static std::unique_ptr<FilterReader> Make(std::unique_ptr<arrow::RecordBatchReader> reader,
                                            const ReadOptions& option);

  FilterReader(std::unique_ptr<arrow::RecordBatchReader> reader, const ReadOptions& option)
      : record_reader_(std::move(reader)), option_(option) {}

  private:
  arrow::Status NextFilteredBatchReader();

  std::unique_ptr<arrow::RecordBatchReader> record_reader_;
  const ReadOptions& option_;
  std::shared_ptr<arrow::RecordBatchReader> current_filtered_batch_reader_;
};
}  // namespace milvus_storage
