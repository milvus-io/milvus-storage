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
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "lance_bridge.h"  // from cpp/src/format/lance/lance-bridge/src/include

namespace milvus_storage::lance {

class LanceTableReader final : public FormatReader, public std::enable_shared_from_this<LanceTableReader> {
  public:
  LanceTableReader(const std::shared_ptr<BlockingDataset> dataset,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const milvus_storage::api::Properties& properties);

  LanceTableReader(const std::string& uri,
                   uint64_t fragment_id,
                   const std::shared_ptr<arrow::Schema>& schema,
                   const milvus_storage::api::Properties& properties);

  [[nodiscard]] arrow::Status open() override;

  // get the row group infos
  [[nodiscard]] arrow::Result<std::vector<RowGroupInfo>> get_row_group_infos() override;

  // get the chunk
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(const int& row_group_index) override;

  // get the chunks
  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int>& rg_indices_in_file) override;

  // take
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;

  // read with range
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> read_with_range(
      const uint64_t& start_offset, const uint64_t& end_offset) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<FormatReader>> clone_reader() override;

  private:
  std::shared_ptr<BlockingDataset> dataset_;
  std::string uri_;
  uint64_t fragment_id_;
  std::shared_ptr<arrow::Schema> schema_;
  milvus_storage::api::Properties properties_;

  uint64_t logical_chunk_rows_;
  std::vector<RowGroupInfo> row_group_infos_;
  std::unique_ptr<BlockingFragmentReader> fragment_reader_;
};

}  // namespace milvus_storage::lance
