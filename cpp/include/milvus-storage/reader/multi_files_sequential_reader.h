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

#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include "file/fragment.h"
#include "storage/space.h"

namespace milvus_storage {

class MultiFilesSequentialReader : public arrow::RecordBatchReader {
  public:
  MultiFilesSequentialReader(arrow::fs::FileSystem& fs,
                             const FragmentVector& fragments,
                             std::shared_ptr<arrow::Schema> schema,
                             const ReadOptions& options);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  arrow::fs::FileSystem& fs_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> files_;

  size_t next_pos_ = 0;
  std::unique_ptr<arrow::RecordBatchReader> curr_reader_;
  std::unique_ptr<parquet::arrow::FileReader>
      holding_file_reader_;  // file reader have to outlive than record batch reader, so we hold here.
  const ReadOptions options_;

  friend FilterQueryRecordReader;
};
}  // namespace milvus_storage
