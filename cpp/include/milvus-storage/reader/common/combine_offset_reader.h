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
#include "milvus-storage/format/parquet/file_reader.h"
#include "milvus-storage/storage/schema.h"
namespace milvus_storage {

// CombineOffsetReader reads records from a reader and fetches corresponding records
// of another file and combines them together.
class CombineOffsetReader : public arrow::RecordBatchReader {
  public:
  static std::unique_ptr<CombineOffsetReader> Make(std::unique_ptr<arrow::RecordBatchReader> scalar_reader,
                                                   std::unique_ptr<ParquetFileReader> vector_reader,
                                                   std::shared_ptr<Schema> schema);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  CombineOffsetReader(std::shared_ptr<arrow::RecordBatchReader> scalar_reader,
                      std::shared_ptr<ParquetFileReader> vector_reader,
                      std::shared_ptr<Schema> schema)
      : scalar_reader_(std::move(scalar_reader)),
        vector_reader_(std::move(vector_reader)),
        schema_(std::move(schema)) {}

  private:
  std::shared_ptr<arrow::RecordBatchReader> scalar_reader_;
  std::shared_ptr<ParquetFileReader> vector_reader_;
  std::shared_ptr<Schema> schema_;
};
}  // namespace milvus_storage
