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
#include "arrow/record_batch.h"
#include "milvus-storage/storage/schema.h"

namespace milvus_storage {

// CombineReader merges scalar fields and vector fields to an entire record.
class CombineReader : public arrow::RecordBatchReader {
  public:
  static std::unique_ptr<CombineReader> Make(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                                             std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                                             std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  CombineReader(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                std::unique_ptr<arrow::RecordBatchReader> vector_reader,
                std::shared_ptr<Schema> schema)
      : scalar_reader_(std::move(scalar_reader)),
        vector_reader_(std::move(vector_reader)),
        schema_(std::move(schema)) {}

  private:
  std::unique_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::unique_ptr<arrow::RecordBatchReader> vector_reader_;
  std::shared_ptr<Schema> schema_;
};
}  // namespace milvus_storage
