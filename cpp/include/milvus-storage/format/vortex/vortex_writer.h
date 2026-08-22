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

#include <optional>

#include <memory>

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/writer_status.h"
#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/column_groups.h"
#include "vortex_bridge.h"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexFileWriter final : public FormatWriter {
  public:
  static arrow::Result<std::unique_ptr<VortexFileWriter>> Open(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                               std::shared_ptr<arrow::Schema> schema,
                                                               const std::string& file_path,
                                                               const api::Properties& properties);

  ~VortexFileWriter() = default;

  VortexFileWriter(VortexFileWriter&& other) noexcept = default;
  VortexFileWriter& operator=(VortexFileWriter&& other) noexcept = default;

  VortexFileWriter(const VortexFileWriter&) = delete;
  VortexFileWriter& operator=(const VortexFileWriter&) = delete;

  [[nodiscard]] arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  [[nodiscard]] arrow::Status Flush() override;

  [[nodiscard]] arrow::Result<api::ColumnGroupFile> Close() override;

  void Abort() noexcept override;

#ifdef BUILD_GTEST
  bool HasBridgeWriterForTest() const { return vx_writer_.has_value(); }
#endif

  private:
  arrow::Status WriteImpl(const std::shared_ptr<arrow::RecordBatch>& record);
  arrow::Status FlushImpl();
  arrow::Result<api::ColumnGroupFile> CloseImpl();

  VortexFileWriter(std::unique_ptr<FileSystemWrapper> fs_holder,
                   VortexWriter vx_writer,
                   std::shared_ptr<arrow::Schema> schema,
                   std::string file_path,
                   api::Properties properties);

  bool closed_;
  std::string file_path_;
  std::unique_ptr<FileSystemWrapper> fs_holder_;
  // optional so Abort() can release the bridge writer immediately. Dropping it
  // drops the Rust ObjectStoreWriterInner, whose Drop calls
  // loon_filesystem_writer_destroy -- which aborts the C++ output stream and
  // therefore the multipart upload. Holding it until this object is destroyed
  // would still release it, just later than the caller asked.
  std::optional<VortexWriter> vx_writer_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;
  WriterStatus writer_status_;

  int64_t written_rows_ = 0;
};
}  // namespace milvus_storage::vortex
