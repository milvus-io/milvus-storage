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
#ifdef BUILD_VORTEX_BRIDGE

#include "milvus-storage/format/vortex/vortex_writer.h"
#include <arrow/c/bridge.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>
#include <string>

#include "milvus-storage/properties.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

VortexFileWriter::VortexFileWriter(std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
                                   std::shared_ptr<ObjectStoreWrapper> fs,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const api::Properties& properties)
    : closed_(false),
      obsw_(std::move(fs)),
      vx_writer_(
          std::move(VortexWriter::Open(*obsw_,
                                       column_group->files[0].path,
                                       GetValueNoError<bool>(properties, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS)))),
      schema_(schema),
      properties_(properties) {}

arrow::Status VortexFileWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  assert(!closed_);
  assert(batch->schema()->Equals(*schema_, false));
  count_ += batch->num_rows();
  bytes_written_ += milvus_storage::GetRecordBatchMemorySize(batch);

  ARROW_ASSIGN_OR_RAISE(auto arrow_struct_array, batch->ToStructArray());

  column_arrays_.emplace_back(arrow_struct_array);
  assert(arrow_struct_array->num_fields() != 0);
  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::Flush() {
  ArrowArray exported_array;
  ArrowSchema exported_schema;
  assert(!closed_);

  for (const auto& struct_array : column_arrays_) {
    ARROW_RETURN_NOT_OK(arrow::ExportArray(*struct_array, &exported_array, &exported_schema));
    vx_writer_.Write(exported_schema, exported_array);
  }

  column_arrays_.clear();
  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::Close() {
  assert(!closed_);

  ARROW_RETURN_NOT_OK(Flush());
  vx_writer_.Close();
  closed_ = true;
  return arrow::Status::OK();
}

}  // namespace milvus_storage::vortex
#endif