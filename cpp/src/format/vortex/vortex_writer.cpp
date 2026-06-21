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

#include "milvus-storage/format/vortex/vortex_writer.h"
#include <arrow/c/bridge.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>
#include <string>
#include <utility>

#include "milvus-storage/properties.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage::vortex {

using namespace milvus_storage::api;

VortexFileWriter::VortexFileWriter(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const std::string& file_path,
                                   const api::Properties& properties)
    : closed_(false),
      file_path_(file_path),
      fs_holder_(std::make_unique<FileSystemWrapper>(fs)),
      vx_writer_(std::move(VortexWriter::Open(
          reinterpret_cast<uint8_t*>(fs_holder_.get()),
          file_path_,
          GetValueNoError<bool>(properties, PROPERTY_WRITER_VORTEX_ENABLE_STATISTICS),
          static_cast<uint32_t>(GetValueNoError<uint64_t>(properties, PROPERTY_WRITER_VORTEX_FORMAT_VERSION)),
          GetValueNoError<uint64_t>(properties, PROPERTY_WRITER_VORTEX_V2_ROW_GROUP_MAX_SIZE)))),
      schema_(std::move(schema)),
      properties_(properties) {}

arrow::Status VortexFileWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  try {
    assert(!closed_);
    assert(batch->schema()->Equals(*schema_, false));

    ARROW_ASSIGN_OR_RAISE(auto arrow_struct_array, batch->ToStructArray());
    assert(arrow_struct_array->num_fields() != 0);
    ArrowArray exported_array;
    ArrowSchema exported_schema;
    ARROW_RETURN_NOT_OK(arrow::ExportArray(*arrow_struct_array, &exported_array, &exported_schema));
    vx_writer_.Write(exported_schema, exported_array);
    written_rows_ += batch->num_rows();
    return arrow::Status::OK();
  } catch (const VortexException& e) {
    closed_ = true;
    return arrow::Status::IOError("Failed to write Vortex file: ", e.what());
  }
}

arrow::Status VortexFileWriter::Flush() {
  try {
    assert(!closed_);
    vx_writer_.Flush();
    return arrow::Status::OK();
  } catch (const VortexException& e) {
    return arrow::Status::IOError("Failed to flush Vortex file: ", e.what());
  }
}

arrow::Result<api::ColumnGroupFile> VortexFileWriter::Close() {
  try {
    assert(!closed_);

    // Close returns the total file size and footer size from WriteSummary
    auto summary = vx_writer_.Close();
    closed_ = true;
    return api::ColumnGroupFile{
        .path = file_path_,
        .start_index = 0,
        .end_index = written_rows_,
        .properties = {{api::kPropertyFileSize, std::to_string(summary.file_size)},
                       {api::kPropertyFooterSize, std::to_string(summary.footer_size)}},
    };
  } catch (const VortexException& e) {
    closed_ = true;
    return arrow::Status::IOError("Failed to close Vortex file: ", e.what());
  }
}

}  // namespace milvus_storage::vortex
