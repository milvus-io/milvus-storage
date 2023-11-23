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

#include "reader/scan_record_reader.h"
#include <memory>
#include "common/macro.h"
#include "common/utils.h"
#include "reader/common/delete_reader.h"
#include "reader/common/filter_reader.h"
#include "reader/common/projection_reader.h"
#include "reader/multi_files_sequential_reader.h"

namespace milvus_storage {

ScanRecordReader::ScanRecordReader(std::shared_ptr<Schema> schema,
                                   std::shared_ptr<ReadOptions> options,
                                   std::shared_ptr<arrow::fs::FileSystem> fs,
                                   const FragmentVector& fragments,
                                   const DeleteFragmentVector& delete_fragments)
    : schema_(schema), options_(options), fs_(fs), fragments_(fragments), delete_fragments_(delete_fragments) {}

std::shared_ptr<arrow::Schema> ScanRecordReader::schema() const {
  auto r = ProjectSchema(schema_->schema(), options_->output_columns());
  if (!r.ok()) {
    return nullptr;
  }
  return r.value();
}
arrow::Status ScanRecordReader::ReadNext(std::shared_ptr<arrow::RecordBatch>* batch) {
  if (reader_ == nullptr) {
    auto res = MakeInnerReader();
    if (!res.ok()) {
      return arrow::Status::UnknownError(res.status().ToString());
    }
    reader_ = res.value();
  }

  return reader_->ReadNext(batch);
}

Result<std::shared_ptr<arrow::RecordBatchReader>> ScanRecordReader::MakeInnerReader() {
  auto reader = std::make_shared<MultiFilesSequentialReader>(fs_, fragments_, schema_->schema(), options_);
  ASSIGN_OR_RETURN_NOT_OK(auto filter_reader, FilterReader::Make(reader, options_));
  auto delete_reader = DeleteMergeReader::Make(filter_reader, schema_->options(), delete_fragments_, options_);
  ASSIGN_OR_RETURN_NOT_OK(auto res, ProjectionReader::Make(schema_->schema(), delete_reader, options_));
  return res;
}
}  // namespace milvus_storage
