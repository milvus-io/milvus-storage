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

#include <arrow/chunked_array.h>

#include "bridgeimpl.hpp"  // from cpp/src/format/vortex/vx-bridge/src/include

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

namespace milvus_storage::vortex {

class VortexFormatReader final : public FormatReader, public std::enable_shared_from_this<VortexFormatReader> {
  public:
  VortexFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                     const std::shared_ptr<arrow::Schema>& schema,
                     const std::string& path,
                     const milvus_storage::api::Properties& properties,
                     const std::vector<std::string>& needed_columns);

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

  // get the row ranges(splits) of the file
  inline std::vector<uint64_t> row_ranges() const { return vxfile_->Splits(); }

  // get the total rows of the file
  inline size_t rows() const { return vxfile_->RowCount(); }

  // get the total memory usage(uncompressed memory) of the file
  uint64_t total_mem_usage();

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> streaming_read(uint64_t row_start,
                                                                                        uint64_t row_end);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::ChunkedArray>> blocking_read(uint64_t row_start, uint64_t row_end);

  private:
  [[nodiscard]] arrow::Result<ArrowArrayStream> read(uint64_t row_start, uint64_t row_end);

  private:
  std::shared_ptr<FileSystemWrapper> fs_holder_;
  std::vector<std::string> proj_cols_;
  std::string path_;
  std::shared_ptr<arrow::Schema> schema_;
  milvus_storage::api::Properties properties_;

  uint64_t logical_chunk_rows_;
  std::vector<RowGroupInfo> row_group_infos_;
  std::unique_ptr<VortexFile> vxfile_;
};

}  // namespace milvus_storage::vortex

#endif  // BUILD_VORTEX_BRIDGE