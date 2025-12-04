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

#include <unordered_map>
#include <string_view>
#include <string>

#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/column_group_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {
using internal::api::ChunkInfo;
using internal::api::RowGroupInfo;
class VortexChunkReader final : public internal::api::ColumnGroupReaderInternal {
  public:
  VortexChunkReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                    const std::shared_ptr<arrow::Schema>& schema,
                    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                    const api::Properties& properties,
                    const std::vector<std::string>& needed_columns);

  ~VortexChunkReader();

  [[nodiscard]] arrow::Result<std::pair<std::vector<ChunkInfo>, std::vector<std::vector<RowGroupInfo>>>> open()
      override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunk(size_t file_index,
                                                                       const std::vector<RowGroupInfo>& row_group_info,
                                                                       const int& rg_index_in_file) override;
  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Table>> get_chunks(
      size_t file_index,
      const std::vector<RowGroupInfo>& row_group_info,
      const std::vector<int>& rg_indices_in_file) override;

  private:
  std::shared_ptr<FileSystemWrapper> fs_holder_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> proj_cols_;
  api::Properties properties_;

  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  std::vector<milvus_storage::api::ColumnGroupFile> cg_files_;
  std::vector<std::shared_ptr<VortexFormatReader>> vortex_readers_;

  uint64_t logical_chunk_rows_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE