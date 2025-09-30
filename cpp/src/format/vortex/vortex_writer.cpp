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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/vortex/vortex_writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "object_store_ffi.h"
#include "object_store_writer_ffi.h"
#include <arrow/c/bridge.h>
#include <arrow/chunked_array.h>
#include <arrow/array.h>
#include <string>

namespace milvus_storage::vortex {

VortexFileWriter::VortexFileWriter(ArrowFileSystemConfig config,
                                   std::shared_ptr<arrow::Schema> schema,
                                   const std::string& file_path,
                                   const api::Properties& properties)
    : fs_config_(std::move(config)),
      schema_(schema),
      file_path_(file_path),
      properties_(properties),
      obj_store_(nullptr) {}

arrow::Status VortexFileWriter::Write(const std::shared_ptr<arrow::RecordBatch> batch) {
  assert(batch->schema()->Equals(*schema_, false));
  count_ += batch->num_rows();
  bytes_written_ += milvus_storage::GetRecordBatchMemorySize(batch);

  ARROW_ASSIGN_OR_RAISE(auto arrow_struct_array, batch->ToStructArray());

  column_arrays_.emplace_back(arrow_struct_array);
  assert(arrow_struct_array->num_fields() != 0);
  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::Flush() {
  ArrowArrayStream stream_reader{};

  if (!obj_store_) {
    auto rc = create_object_store(fs_config_.cloud_provider.c_str(), fs_config_.address.c_str(),
                                  fs_config_.access_key_id.c_str(), fs_config_.access_key_value.c_str(),
                                  fs_config_.region.c_str(), fs_config_.bucket_name.c_str(), &obj_store_);

    if (rc != C_SUCCESS) {
      return arrow::Status::Invalid("Failed to init the object store in rust. rc: " + std::to_string(rc));
    }
  }

  std::shared_ptr<arrow::ChunkedArray> chunkarray = std::make_shared<arrow::ChunkedArray>(column_arrays_);
  ARROW_RETURN_NOT_OK(ExportChunkedArray(chunkarray, &stream_reader));

  auto rc = write_array_stream(obj_store_, reinterpret_cast<uint8_t*>(&stream_reader), file_path_.c_str());
  if (rc != C_SUCCESS) {
    return arrow::Status::Invalid("Failed to write array stream in rust. rc: " + std::to_string(rc));
  }

  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::Close() {
  free_object_store_wrapper(obj_store_);
  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::AppendKVMetadata(const std::string& key, const std::string& value) {
  // nothing to do
  return arrow::Status::OK();
}

arrow::Status VortexFileWriter::AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata) {
  // nothing to do
  return arrow::Status::OK();
}

}  // namespace milvus_storage::vortex
#endif