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

#ifdef BUILD_LANCE_BRIDGE
#ifdef BUILD_GTEST

#pragma once

#include <memory>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "lance_bridge.h"  // from cpp/src/format/lance/lance-bridge/src/include

namespace milvus_storage::lance {

/**
 * Current writer won't used, except test
 */
class LanceTableWriter final : public FormatWriter {
  public:
  LanceTableWriter(const std::string& base_path,
                   std::shared_ptr<arrow::Schema> schema,
                   const api::Properties& properties);

  ~LanceTableWriter() = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Result<api::ColumnGroupFile> Close() override;

  private:
  bool closed_;
  std::string base_path_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
  std::unique_ptr<BlockingDataset> dataset_;
  std::vector<uint64_t> origin_fids_;
  int64_t written_rows_ = 0;
};
}  // namespace milvus_storage::lance

#endif  // BUILD_GTEST
#endif  // BUILD_LANCE_BRIDGE
