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
#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexFileWriter : public internal::api::ColumnGroupWriter {
  public:
  VortexFileWriter(std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
                   std::shared_ptr<arrow::Schema> schema,
                   const api::Properties& properties);

  ~VortexFileWriter() = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Status Close() override;

  arrow::Status AppendKVMetadata(const std::string& key, const std::string& value) override;

  int64_t count() const { return count_; }
  int64_t bytes_written() const { return bytes_written_; }

  private:
  bool closed_;
  ObjectStoreWrapper obsw_;
  VortexWriter vx_writer_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;

  std::vector<std::shared_ptr<arrow::Array>> column_arrays_;

  int64_t count_ = 0;
  int64_t bytes_written_ = 0;
};
}  // namespace milvus_storage::vortex
#endif