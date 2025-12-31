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
#include "milvus-storage/format/column_group_writer.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "lance_bridge.hpp"  // from cpp/src/format/lance/lance-bridge/src/include

namespace milvus_storage::lance {

class LanceFragmentWriter final : public api::ColumnGroupWriter {
  public:
  LanceFragmentWriter(std::shared_ptr<api::ColumnGroup> column_group,
                      std::shared_ptr<arrow::Schema> schema,
                      const api::Properties& properties);

  ~LanceFragmentWriter() = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Status Close() override;

  uint64_t written_rows() const override { return written_rows_; }

  private:
  bool closed_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;

  std::string base_path_;
  std::unique_ptr<BlockingDataset> dataset_;
  int64_t written_rows_ = 0;
};
}  // namespace milvus_storage::lance