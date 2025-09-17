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
#pragma once

#include <memory>
#include "milvus-storage/common/config.h"
#include "milvus-storage/writer.h"
#include "milvus-storage/manifest.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/filesystem/fs.h"
#include "object_store_ffi.h"

namespace milvus_storage::vortex {

class VortexFileWriter : public internal::api::ColumnGroupWriter {
  public:
  VortexFileWriter(ArrowFileSystemConfig config,
                   std::shared_ptr<arrow::Schema> schema,
                   const std::string& file_path,
                   const api::Properties& properties);

  ~VortexFileWriter() = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Status Close() override;

  arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) override;

  arrow::Status AddUserMetadata(const std::vector<std::pair<std::string, std::string>>& metadata);

  int64_t count() const { return count_; }
  int64_t bytes_written() const { return bytes_written_; }

  private:
  ArrowFileSystemConfig fs_config_;
  std::shared_ptr<arrow::Schema> schema_;
  const std::string file_path_;
  api::Properties properties_;

  ObjectStoreWrapper* obj_store_;
  std::vector<std::shared_ptr<arrow::Array>> column_arrays_;

  int64_t count_ = 0;
  int64_t bytes_written_ = 0;
};
}  // namespace milvus_storage::vortex
#endif