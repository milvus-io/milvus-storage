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

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/column_groups.h"
#include "vortex_bridge.h"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexFileWriter final : public FormatWriter {
  public:
  VortexFileWriter(std::shared_ptr<arrow::fs::FileSystem> fs,
                   std::shared_ptr<arrow::Schema> schema,
                   const std::string& file_path,
                   const api::Properties& properties);

  ~VortexFileWriter() = default;

  [[nodiscard]] arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  [[nodiscard]] arrow::Status Flush() override;

  [[nodiscard]] arrow::Result<api::ColumnGroupFile> Close() override;

  private:
  bool closed_;
  std::string file_path_;
  std::unique_ptr<FileSystemWrapper> fs_holder_;
  VortexWriter vx_writer_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;

  std::vector<std::shared_ptr<arrow::Array>> column_arrays_;

  int64_t written_rows_ = 0;
};
}  // namespace milvus_storage::vortex