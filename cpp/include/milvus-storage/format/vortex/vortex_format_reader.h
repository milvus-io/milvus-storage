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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/chunked_array.h>
#include <arrow/c/abi.h>

#include "milvus-storage/common/config.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

namespace milvus_storage::vortex {

class VortexFile;
struct VortexReadPlan;
namespace ffi {
struct CoalescingWindow;
}  // namespace ffi
extern const ffi::CoalescingWindow kSmallCoalescingWindow;
namespace expr {
class Expr;
}  // namespace expr

class VortexFormatReader final : public FormatReader, public std::enable_shared_from_this<VortexFormatReader> {
  public:
  struct MetaTrait {
    struct Payload {
      std::shared_ptr<FileSystemWrapper> fs_holder;
      std::shared_ptr<VortexFile> vxfile;
      std::vector<uint64_t> row_ranges;
      uint64_t row_count = 0;
      uint64_t memory_usage = 0;
    };

    using Metadata = FormatReaderMetadata<Payload>;
    using MetadataPtr = std::shared_ptr<const Metadata>;

    static std::string cache_key(const api::ColumnGroupFile& file);

    static arrow::Result<MetadataPtr> load_metadata(const api::ColumnGroupFile& file,
                                                    const api::Properties& properties,
                                                    const KeyRetriever& key_retriever);

    static arrow::Result<std::shared_ptr<VortexFormatReader>> create_from_metadata(
        MetadataPtr metadata,
        const api::ColumnGroupFile& file,
        const api::Properties& properties,
        const std::shared_ptr<arrow::Schema>& read_schema,
        const std::vector<std::string>& needed_columns,
        const std::string& predicate);
  };

  VortexFormatReader(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                     const std::shared_ptr<arrow::Schema>& schema,
                     const std::string& path,
                     const milvus_storage::api::Properties& properties,
                     const std::vector<std::string>& needed_columns,
                     uint64_t file_size = 0,
                     uint64_t footer_size = 0);
  // Defined in the cpp because vxfile_ is a unique_ptr to a forward-declared VortexFile.
  ~VortexFormatReader() override;

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

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  void set_predicate(const std::string& predicate) override;

  // get the row ranges(splits) of the file
  std::vector<uint64_t> row_ranges() const;

  // get the total rows of the file
  size_t rows() const;

  // get the total memory usage(uncompressed memory) of the file
  uint64_t total_mem_usage();

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> streaming_read(
      uint64_t row_start, uint64_t row_end, const ffi::CoalescingWindow& coalescing_window);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::ChunkedArray>> blocking_read(
      uint64_t row_start, uint64_t row_end, const ffi::CoalescingWindow& coalescing_window);

  // Vortex-specific extension: execute a planner-generated scan plan while keeping
  // the existing FormatReader interfaces unchanged.
  [[nodiscard]] arrow::Result<ArrowArrayStream> read_with_plan(const VortexReadPlan& plan);

  // Vortex-specific extension: execute a planner-generated scan plan and return
  // the original file-local row indices matching the plan.
  [[nodiscard]] arrow::Result<ArrowArrayStream> read_row_ids_with_plan(const VortexReadPlan& plan);

  private:
  [[nodiscard]] static arrow::Result<std::optional<bool>> parse_split_row_indices_override(const std::string& mode);

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::Schema>> output_schema() const;

  VortexFormatReader(MetaTrait::MetadataPtr metadata,
                     uint64_t file_size,
                     uint64_t footer_size,
                     milvus_storage::api::Properties properties,
                     uint64_t logical_chunk_rows,
                     std::vector<RowGroupInfo> row_group_infos,
                     const std::shared_ptr<arrow::Schema>& read_schema,
                     const std::vector<std::string>& needed_columns);

  [[nodiscard]] arrow::Result<ArrowArrayStream> read(uint64_t row_start,
                                                     uint64_t row_end,
                                                     const ffi::CoalescingWindow& coalescing_window);

  private:
  std::shared_ptr<FileSystemWrapper> fs_holder_;
  std::vector<std::string> proj_cols_;
  std::string path_;
  std::shared_ptr<arrow::Schema> read_schema_;
  milvus_storage::api::Properties properties_;
  uint64_t file_size_ = 0;    ///< Pre-known file size to skip S3 HEAD requests
  uint64_t footer_size_ = 0;  ///< Pre-known footer size for single-IO footer read
  std::optional<bool> split_row_indices_;

  std::shared_ptr<arrow::Schema> file_schema_;  // always derived from file in open()

  uint64_t logical_chunk_rows_ = 0;
  std::vector<RowGroupInfo> row_group_infos_;
  std::shared_ptr<VortexFile> vxfile_;
  std::unique_ptr<expr::Expr> parsed_predicate_;
};

}  // namespace milvus_storage::vortex
