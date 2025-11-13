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
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

namespace milvus_storage::vortex {

class VortexChunkReader final : public internal::api::ColumnGroupReader {
  public:
  VortexChunkReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                    const std::shared_ptr<arrow::Schema>& schema,
                    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                    const api::Properties& properties,
                    const std::vector<std::string>& needed_columns);

  ~VortexChunkReader();
  [[nodiscard]] arrow::Status open() override;

  [[nodiscard]] size_t total_number_of_chunks() const override;

  [[nodiscard]] size_t total_rows() const override;

  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> take(
      const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_size(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;

  private:
  // Information of a row group
  struct ChunkInfo {
    size_t belong_which_file;    // which file this row group belongs to
    uint64_t global_row_offset;  // the starting row index of this row group in the whole chunk reader
    uint64_t row_index_in_file;  // the starting row index of this row group in its file
    uint64_t number_of_rows;     // number of rows in this row group

    uint64_t avg_memory_usage;  // average memory usage of this row group
  };

  private:
  std::shared_ptr<FileSystemWrapper> fs_holder_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> proj_cols_;
  api::Properties properties_;

  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  std::vector<std::string> paths_;
  std::vector<std::unique_ptr<VortexFormatReader>> vxfiles_;

  uint64_t logical_chunk_rows_;
  std::vector<ChunkInfo> rginfos_;
  std::vector<uint64_t> offsets_in_paths_;
  uint64_t total_rows_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE