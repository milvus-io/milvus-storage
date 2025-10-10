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

#include <arrow/type.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>
#include "milvus-storage/storage/options.h"

namespace milvus_storage {
class ProjectionReader : public arrow::RecordBatchReader {
  public:
  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  static arrow::Result<std::unique_ptr<arrow::RecordBatchReader>> Make(std::shared_ptr<arrow::Schema> schema,
                                                                       std::unique_ptr<arrow::RecordBatchReader> reader,
                                                                       const ReadOptions& options);

  ProjectionReader(std::shared_ptr<arrow::Schema> schema,
                   std::unique_ptr<arrow::RecordBatchReader> reader,
                   const ReadOptions& options);

  private:
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  const ReadOptions options_;
  std::shared_ptr<arrow::Schema> schema_;
};
}  // namespace milvus_storage
