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

#include "arrow/filesystem/filesystem.h"
#include "milvus-storage/common/metadata.h"
#include "parquet/arrow/reader.h"
#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/filesystem/fs.h"
#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexChunkReader final : public internal::api::ColumnGroupReader {
  public:
  VortexChunkReader(std::shared_ptr<ObjectStoreWrapper> fs,
                    std::shared_ptr<arrow::Schema> schema,
                    const std::vector<std::string>& paths,
                    const std::vector<std::string>& needed_columns,
                    const api::Properties& properties);

  ~VortexChunkReader();
  [[nodiscard]] arrow::Status open() override;

  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(
      const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_size(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<int64_t> get_chunk_rows(int64_t chunk_index) override;

  private:
  arrow::Result<std::vector<std::vector<int64_t>>> calc_ridxs_in_chunks(const std::vector<int64_t>& row_indices);

  private:
  std::shared_ptr<ObjectStoreWrapper> obsw_;
  const size_t number_of_chunks_;

  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> proj_cols_;
  api::Properties properties_;

  std::vector<std::string> paths_;
  std::vector<std::unique_ptr<VortexFormatReader>> vxfiles_;
  std::vector<size_t> idx_offsets_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE