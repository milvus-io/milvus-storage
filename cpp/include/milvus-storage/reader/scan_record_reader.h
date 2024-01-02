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
#include "file/delete_fragment.h"
#include "file/fragment.h"
#include "reader/multi_files_sequential_reader.h"
namespace milvus_storage {
class ScanRecordReader : public arrow::RecordBatchReader {
  public:
  ScanRecordReader(std::shared_ptr<Schema> schema,
                   const ReadOptions& options,
                   std::shared_ptr<arrow::fs::FileSystem> fs,
                   const FragmentVector& fragments,
                   const DeleteFragmentVector& delete_fragments);

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) override;

  private:
  Result<std::shared_ptr<arrow::RecordBatchReader>> MakeInnerReader();

  std::shared_ptr<Schema> schema_;
  const ReadOptions options_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  const FragmentVector fragments_;
  const DeleteFragmentVector delete_fragments_;
  std::shared_ptr<arrow::RecordBatchReader> reader_;
};

}  // namespace milvus_storage
