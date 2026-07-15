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

#include "milvus-storage/format/column_group_reader.h"

#include <algorithm>
#include <future>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage::api {

using milvus_storage::RowGroupInfo;
using ChunkRBMapResult = arrow::Result<std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>>>;

namespace {

arrow::Result<ColumnMemorySizesPtr> BuildColumnMemorySizes(const arrow::Schema& file_schema,
                                                           const std::vector<uint64_t>& sizes) {
  if (sizes.empty()) {
    return ColumnMemorySizesPtr{};
  }
  if (sizes.size() != static_cast<size_t>(file_schema.num_fields())) {
    return arrow::Status::Invalid("Column memory size count does not match the file schema");
  }

  auto column_memory_sizes = std::make_shared<ColumnMemorySizes>();
  column_memory_sizes->reserve(sizes.size());
  for (int field_index = 0; field_index < file_schema.num_fields(); ++field_index) {
    const auto& field_name = file_schema.field(field_index)->name();
    if (!column_memory_sizes->emplace(field_name, sizes[field_index]).second) {
      return arrow::Status::Invalid("Duplicate field name in file schema: ", field_name);
    }
  }
  return std::static_pointer_cast<const ColumnMemorySizes>(column_memory_sizes);
}

}  // namespace

template <typename ReaderT>
class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever,
                        const milvus_storage::MetadataCache& cache,
                        const std::string& predicate = "");

  ~ColumnGroupReaderImpl() override = default;

  [[nodiscard]] arrow::Status open() override;
  [[nodiscard]] folly::SemiFuture<arrow::Status> open_async();
  [[nodiscard]] size_t total_number_of_chunks() const override;
  [[nodiscard]] size_t total_rows() const override;
  [[nodiscard]] arrow::Result<std::vector<int64_t>> get_chunk_indices(const std::vector<int64_t>& row_indices) override;

  [[nodiscard]] arrow::Result<std::shared_ptr<arrow::RecordBatch>> get_chunk(int64_t chunk_index) override;

  [[nodiscard]] arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> get_chunks(
      const std::vector<int64_t>& chunk_indices, size_t parallelism = 1) override;

  [[nodiscard]] arrow::Result<uint64_t> get_chunk_estimated_size(int64_t chunk_index) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_column_estimated_size(int64_t chunk_index, int col_idx) override;
  [[nodiscard]] arrow::Result<uint64_t> get_chunk_rows(int64_t chunk_index) override;

  [[nodiscard]] const ChunkInfo& get_chunk_info(int64_t chunk_index) const override;
  [[nodiscard]] folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> get_chunks_async(
      const ChunkTask& task) override;

  [[nodiscard]] std::shared_ptr<arrow::Schema> get_schema() const override;

  private:
  struct OpenedFile {
    size_t file_index;
    ColumnGroupFile file;
    std::shared_ptr<ReaderT> reader;
    std::vector<RowGroupInfo> row_group_infos;
    std::shared_ptr<arrow::Schema> file_schema;
  };

  [[nodiscard]] arrow::Status append_file_metadata(size_t file_idx,
                                                   const ColumnGroupFile& cg_file,
                                                   const std::vector<RowGroupInfo>& row_group_in_file,
                                                   size_t& rows_in_all_files);

  ChunkRBMapResult read_chunks_from_files(const std::vector<int64_t>& task_indices);
  arrow::Result<std::shared_ptr<ReaderT>> open_reader_for_file(size_t file_index);
  folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>> open_reader_for_file_async(size_t file_index);

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  milvus_storage::MetadataCache cache_;
  std::string predicate_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  std::shared_ptr<arrow::Schema> file_schema_;
  size_t total_rows_ = 0;

  std::vector<std::shared_ptr<ReaderT>> format_readers_;
  bool opened_ = false;
};  // ColumnGroupReaderImpl

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

template <typename ReaderT>
ColumnGroupReaderImpl<ReaderT>::ColumnGroupReaderImpl(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<api::ColumnGroup>& column_group,
    const api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
    const milvus_storage::MetadataCache& cache,
    const std::string& predicate)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      cache_(cache),
      predicate_(predicate) {}

template <typename ReaderT>
arrow::Status ColumnGroupReaderImpl<ReaderT>::append_file_metadata(size_t file_idx,
                                                                   const ColumnGroupFile& cg_file,
                                                                   const std::vector<RowGroupInfo>& row_group_in_file,
                                                                   size_t& rows_in_all_files) {
  row_group_infos_[file_idx] = row_group_in_file;
  if (row_group_in_file.empty()) {
    return arrow::Status::OK();
  }

  auto current_file_schema = format_readers_[file_idx] ? format_readers_[file_idx]->get_schema() : nullptr;
  auto build_column_memory_sizes = [&](const std::vector<uint64_t>& sizes) -> ColumnMemorySizesPtr {
    if (sizes.empty()) {
      return nullptr;
    }
    if (!current_file_schema) {
      LOG_STORAGE_DEBUG_ << "Column memory sizes are unavailable because the file schema is missing"
                         << ", path=" << cg_file.path;
      return nullptr;
    }
    auto result = BuildColumnMemorySizes(*current_file_schema, sizes);
    if (!result.ok()) {
      // Column estimates are optional. Preserve the total chunk estimate and
      // normal reads; the public column-estimate API returns NotImplemented.
      LOG_STORAGE_DEBUG_ << "Column memory sizes are unavailable"
                         << ", path=" << cg_file.path << ", status=" << result.status().ToString();
      return nullptr;
    }
    return std::move(result).ValueOrDie();
  };

  size_t rows_in_file = 0;
  if ((cg_file.start_index != 0 || cg_file.end_index != row_group_in_file.back().end_offset)) {
    // A manifest file may expose only a subrange of its physical contents.
    // Intersect that range with row groups to produce logical chunk boundaries.
    const auto& start_index = cg_file.start_index;
    const auto& end_index = cg_file.end_index;

    assert(start_index >= 0 && end_index > 0 && start_index < end_index);

    for (size_t j = 0; j < row_group_in_file.size(); ++j) {
      size_t rg_start = row_group_in_file[j].start_offset;
      size_t rg_end = row_group_in_file[j].end_offset;

      size_t overlap_start = std::max(static_cast<size_t>(start_index), rg_start);
      size_t overlap_end = std::min(static_cast<size_t>(end_index), rg_end);

      if (overlap_start < overlap_end) {
        const auto overlap_rows = overlap_end - overlap_start;
        const auto row_group_rows = rg_end - rg_start;
        uint64_t chunk_memory_size = 0;
        if (row_group_in_file[j].memory_size_available) {
          chunk_memory_size = static_cast<uint64_t>(static_cast<unsigned __int128>(row_group_in_file[j].memory_size) *
                                                    overlap_rows / row_group_rows);
        }
        ColumnMemorySizesPtr column_memory_sizes;
        if (row_group_in_file[j].memory_size_available && !row_group_in_file[j].column_memory_sizes.empty()) {
          auto scaled_sizes = DistributeMemorySizes(chunk_memory_size, row_group_in_file[j].column_memory_sizes);
          if (scaled_sizes.ok()) {
            column_memory_sizes = build_column_memory_sizes(*scaled_sizes);
          } else {
            LOG_STORAGE_DEBUG_ << "Column memory size scaling is unavailable"
                               << ", path=" << cg_file.path << ", status=" << scaled_sizes.status().ToString();
          }
        }

        rows_in_file += overlap_rows;
        // global_row_end is exclusive across the whole column group and powers
        // the row-to-chunk binary search used by public readers.
        chunk_infos_.emplace_back(ChunkInfo{
            .file_index = file_idx,
            .row_offset_in_row_group = overlap_start - rg_start,
            .row_offset_in_file = overlap_start,
            .number_of_rows = overlap_rows,
            .row_group_index_in_file = j,
            .global_row_end = rows_in_all_files + rows_in_file,
            .avg_memory_size = chunk_memory_size,
            .column_memory_sizes = std::move(column_memory_sizes),
            .memory_size_available = row_group_in_file[j].memory_size_available,
        });
      }
    }
  } else {
    for (size_t j = 0; j < row_group_in_file.size(); ++j) {
      rows_in_file += (row_group_in_file[j].end_offset - row_group_in_file[j].start_offset);
      chunk_infos_.emplace_back(ChunkInfo{
          .file_index = file_idx,
          .row_offset_in_row_group = 0,
          .row_offset_in_file = row_group_in_file[j].start_offset,
          .number_of_rows = row_group_in_file[j].end_offset - row_group_in_file[j].start_offset,
          .row_group_index_in_file = j,
          .global_row_end = rows_in_all_files + rows_in_file,
          .avg_memory_size = row_group_in_file[j].memory_size,
          .column_memory_sizes = row_group_in_file[j].memory_size_available
                                     ? build_column_memory_sizes(row_group_in_file[j].column_memory_sizes)
                                     : nullptr,
          .memory_size_available = row_group_in_file[j].memory_size_available,
      });
    }
  }

  rows_in_all_files += rows_in_file;
  return arrow::Status::OK();
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<ReaderT>> ColumnGroupReaderImpl<ReaderT>::open_reader_for_file(size_t file_index) {
  if (file_index >= column_group_->files.size()) {
    return arrow::Status::Invalid("Column group file index out of range: ", file_index,
                                  " >= ", column_group_->files.size());
  }

  auto file = column_group_->files[file_index];
  if (!cache_.enabled()) {
    ARROW_ASSIGN_OR_RAISE(auto reader, FormatReader::create(schema_, column_group_->format, file, properties_,
                                                            needed_columns_, key_retriever_));
    if (!predicate_.empty()) {
      ARROW_RETURN_NOT_OK(reader->set_predicate(predicate_));
    }
    auto typed_reader = std::dynamic_pointer_cast<ReaderT>(reader);
    if (!typed_reader) {
      return arrow::Status::Invalid("FormatReader::create returned incompatible reader for format: ",
                                    column_group_->format);
    }
    return typed_reader;
  } else {
    auto key = ReaderT::MetaTrait::cache_key(file);
    ARROW_ASSIGN_OR_RAISE(auto metadata, cache_.get<ReaderT>()->get_or_open(key, [this, file]() {
      return FormatReader::load_metadata<ReaderT>(file, properties_, key_retriever_);
    }));
    return FormatReader::create_from_metadata<ReaderT>(metadata, file, schema_, needed_columns_, predicate_);
  }

  return arrow::Status::Invalid("Unreachable code");
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>> ColumnGroupReaderImpl<ReaderT>::open_reader_for_file_async(
    size_t file_index) {
  if (file_index >= column_group_->files.size()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<ReaderT>>(arrow::Status::Invalid(
        "Column group file index out of range: ", file_index, " >= ", column_group_->files.size())));
  }

  auto file = column_group_->files[file_index];
  if (!cache_.enabled()) {
    // Without metadata caching, create and open a fresh stateful format reader.
    return FormatReader::create_async(schema_, column_group_->format, file, properties_, needed_columns_,
                                      key_retriever_)
        .deferValue([predicate = predicate_,
                     format = column_group_->format](arrow::Result<std::shared_ptr<FormatReader>>&& reader_result)
                        -> arrow::Result<std::shared_ptr<ReaderT>> {
          ARROW_ASSIGN_OR_RAISE(auto reader, std::move(reader_result));
          if (!predicate.empty()) {
            ARROW_RETURN_NOT_OK(reader->set_predicate(predicate));
          }
          auto typed_reader = std::dynamic_pointer_cast<ReaderT>(reader);
          if (!typed_reader) {
            return arrow::Status::Invalid("FormatReader::create_async returned incompatible reader for format: ",
                                          format);
          }
          return typed_reader;
        });
  }

  if constexpr (FormatReaderWithAsyncMetadata<ReaderT>) {
    // Share only immutable metadata across tasks, then apply projection and
    // predicate to a new stateful reader for this file operation.
    auto typed_cache = cache_.get<ReaderT>();
    auto key = ReaderT::MetaTrait::cache_key(file);
    return typed_cache
        ->get_or_open_async(key,
                            [file, properties = properties_, key_retriever = key_retriever_]() {
                              return ReaderT::MetaTrait::load_metadata_async(file, properties, key_retriever);
                            })
        .deferValue([file, read_schema = schema_, needed_columns = needed_columns_,
                     predicate = predicate_](arrow::Result<typename ReaderT::MetaTrait::MetadataPtr>&& metadata_result)
                        -> arrow::Result<std::shared_ptr<ReaderT>> {
          ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(metadata_result));
          return FormatReader::create_from_metadata<ReaderT>(std::move(metadata), file, read_schema, needed_columns,
                                                             predicate);
        });
  }

  // Formats without an async metadata loader keep the synchronous cache path;
  // constructing this ready future may therefore block.
  return folly::makeSemiFuture(open_reader_for_file(file_index));
}

template <typename ReaderT>
arrow::Status ColumnGroupReaderImpl<ReaderT>::open() {
  const auto& cg_files = column_group_->files;

  size_t rows_in_all_files = 0;
  row_group_infos_.clear();
  row_group_infos_.resize(cg_files.size());
  file_schema_.reset();
  format_readers_.clear();
  format_readers_.resize(cg_files.size());
  chunk_infos_.clear();

  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto& cg_file = cg_files[file_idx];

    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", file_idx, cg_file.path));
    }

    ARROW_ASSIGN_OR_RAISE(format_readers_[file_idx], open_reader_for_file(file_idx));
    ARROW_ASSIGN_OR_RAISE(auto row_group_in_file, format_readers_[file_idx]->get_row_group_infos());
    if (!file_schema_) {
      file_schema_ = format_readers_[file_idx]->get_schema();
    }
    ARROW_RETURN_NOT_OK(append_file_metadata(file_idx, cg_file, row_group_in_file, rows_in_all_files));
  }

  total_rows_ = rows_in_all_files;
  opened_ = true;
  return arrow::Status::OK();
}

template <typename ReaderT>
folly::SemiFuture<arrow::Status> ColumnGroupReaderImpl<ReaderT>::open_async() {
  const auto& cg_files = column_group_->files;
  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    const auto& cg_file = cg_files[file_idx];
    if (cg_file.start_index < 0 || cg_file.end_index < 0 || cg_file.start_index >= cg_file.end_index) {
      return folly::makeSemiFuture(arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", file_idx, cg_file.path)));
    }
  }

  std::vector<folly::SemiFuture<arrow::Result<OpenedFile>>> futures;
  futures.reserve(cg_files.size());
  // Create every file-open future before fan-in. Whether work overlaps depends
  // on each format's async factory; ready-future fallbacks may still run inline.
  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto cg_file = cg_files[file_idx];
    auto future = open_reader_for_file_async(file_idx).deferValue(
        [file_idx, cg_file](arrow::Result<std::shared_ptr<ReaderT>>&& reader_result) -> arrow::Result<OpenedFile> {
          ARROW_ASSIGN_OR_RAISE(auto reader, std::move(reader_result));
          ARROW_ASSIGN_OR_RAISE(auto row_group_infos, reader->get_row_group_infos());
          auto file_schema = reader->get_schema();
          return OpenedFile{file_idx, cg_file, std::move(reader), std::move(row_group_infos), std::move(file_schema)};
        });
    futures.push_back(std::move(future));
  }

  // File results stay tagged with their manifest index so shared reader state is
  // assembled deterministically after the fan-in.
  return folly::collectAll(std::move(futures))
      .deferValue([this, file_count = cg_files.size()](auto&& open_results) -> arrow::Status {
        chunk_infos_.clear();
        row_group_infos_.clear();
        row_group_infos_.resize(file_count);
        file_schema_.reset();
        format_readers_.clear();
        format_readers_.resize(file_count);

        size_t rows_in_all_files = 0;
        for (auto& try_result : open_results) {
          if (try_result.hasException()) {
            return arrow::Status::IOError(try_result.exception().what().toStdString());
          }
          ARROW_ASSIGN_OR_RAISE(auto opened_file, std::move(try_result.value()));
          if (!file_schema_ && opened_file.file_schema) {
            file_schema_ = std::move(opened_file.file_schema);
          }
          format_readers_[opened_file.file_index] = std::move(opened_file.reader);
          ARROW_RETURN_NOT_OK(append_file_metadata(opened_file.file_index, opened_file.file,
                                                   opened_file.row_group_infos, rows_in_all_files));
        }

        total_rows_ = rows_in_all_files;
        opened_ = true;
        return arrow::Status::OK();
      });
}

template <typename ReaderT>
size_t ColumnGroupReaderImpl<ReaderT>::total_number_of_chunks() const {
  assert(opened_);
  return chunk_infos_.size();
}

template <typename ReaderT>
size_t ColumnGroupReaderImpl<ReaderT>::total_rows() const {
  assert(opened_);
  return total_rows_;
}

template <typename ReaderT>
arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl<ReaderT>::get_chunk_indices(
    const std::vector<int64_t>& row_indices) {
  assert(opened_);
  // Map logical column-group rows through exclusive chunk ends, preserving the
  // first occurrence of every touched chunk.
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(chunk_infos_.begin(), chunk_infos_.end(), row_index,
                               [](int64_t a, const ChunkInfo& b) { return a < b.global_row_end; });
    auto chunk_index = std::distance(chunk_infos_.begin(), it);
    if (chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid(fmt::format("Row index out of range: {}", row_index));
    }

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl<ReaderT>::get_chunk(int64_t chunk_index) {
  assert(opened_);
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  if (!format_readers_[chunk_info.file_index]) {
    ARROW_ASSIGN_OR_RAISE(format_readers_[chunk_info.file_index], open_reader_for_file(chunk_info.file_index));
  }
  ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[chunk_info.file_index]->get_chunk(chunk_info.row_group_index_in_file));

  // With predicate, Vortex's WithRowRange + WithFilter already produced the
  // correct subset; skip slicing since filtered row counts don't match
  // pre-filter chunk metadata.
  if (predicate_.empty() && (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != rb->num_rows())) {
    rb = rb->Slice(chunk_info.row_offset_in_row_group, chunk_info.number_of_rows);
  }

  return rb;
}

static std::vector<std::vector<int64_t>> split_chunks(const std::vector<int64_t>& sorted_chunk_indices,
                                                      uint64_t parallel_degree) {
  assert(!sorted_chunk_indices.empty());

#ifndef NDEBUG
  for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
    assert(sorted_chunk_indices[i] > sorted_chunk_indices[i - 1]);
  }
#endif

  uint64_t actual_parallel_degree = std::min(parallel_degree, static_cast<uint64_t>(sorted_chunk_indices.size()));

  if (actual_parallel_degree == 0) {
    actual_parallel_degree = 1;
  }

  auto create_continuous_blocks = [&](size_t max_block_size = SIZE_MAX) {
    std::vector<std::vector<int64_t>> continuous_blocks;
    int64_t current_start = sorted_chunk_indices[0];
    int64_t current_count = 1;

    for (size_t i = 1; i < sorted_chunk_indices.size(); ++i) {
      int64_t next_chunk = sorted_chunk_indices[i];

      if (next_chunk == current_start + current_count && current_count < max_block_size) {
        current_count++;
        continue;
      }
      std::vector<int64_t> block(current_count);
      std::iota(block.begin(), block.end(), current_start);
      continuous_blocks.emplace_back(block);
      current_start = next_chunk;
      current_count = 1;
    }

    if (current_count > 0) {
      std::vector<int64_t> block(current_count);
      std::iota(block.begin(), block.end(), current_start);
      continuous_blocks.emplace_back(block);
    }
    return continuous_blocks;
  };

  if (sorted_chunk_indices.size() <= actual_parallel_degree) {
    return create_continuous_blocks();
  }

  size_t avg_block_size = (sorted_chunk_indices.size() + actual_parallel_degree - 1) / actual_parallel_degree;

  return create_continuous_blocks(avg_block_size);
}

template <typename ReaderT>
ChunkRBMapResult ColumnGroupReaderImpl<ReaderT>::read_chunks_from_files(const std::vector<int64_t>& task_indices) {
  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  std::vector<std::vector<int64_t>> chunk_idxs_in_files(column_group_->files.size());

  // Grouping row groups by file
  for (int64_t chunk_index : task_indices) {
    if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
      return arrow::Status::Invalid(
          fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
    }

    const auto& chunk_info = chunk_infos_[chunk_index];
    chunk_idxs_in_files[chunk_info.file_index].emplace_back(chunk_index);
  }

  // Read with range and fill chunk_rb_map
  for (size_t file_idx = 0; file_idx < chunk_idxs_in_files.size(); ++file_idx) {
    const auto& chunk_idxs = chunk_idxs_in_files[file_idx];
    if (chunk_idxs.empty()) {
      continue;
    }

    std::vector<std::pair<uint64_t, uint64_t>> ranges_in_file;

    // generate ranges_in_file and combine the range
    for (int64_t chunk_index : chunk_idxs) {
      const auto& chunk_info = chunk_infos_[chunk_index];
      if (ranges_in_file.empty()) {
        ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                    chunk_info.row_offset_in_file + chunk_info.number_of_rows);
      } else {
        auto& last_range = ranges_in_file.back();

        // won't be overlay in same file
        assert(chunk_info.row_offset_in_file >= last_range.second);
        if (chunk_info.row_offset_in_file == last_range.second) {
          last_range.second = chunk_info.row_offset_in_file + chunk_info.number_of_rows;
        } else {
          ranges_in_file.emplace_back(chunk_info.row_offset_in_file,
                                      chunk_info.row_offset_in_file + chunk_info.number_of_rows);
        }
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto reader, open_reader_for_file(file_idx));
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs_in_file;
    for (auto& range : ranges_in_file) {
      ARROW_ASSIGN_OR_RAISE(auto rbreader, reader->read_with_range(range.first, range.second));
      ARROW_ASSIGN_OR_RAISE(auto rbs, rbreader->ToRecordBatches());
      std::move(rbs.begin(), rbs.end(), std::back_inserter(rbs_in_file));
    }

    // generate chunk_rb_map
    size_t rbs_idx = 0;
    size_t rbs_offset = 0;
    for (long long chunk_idx : chunk_idxs) {
      const auto& chunk_info = chunk_infos_[chunk_idx];
      if (UNLIKELY(((rbs_in_file[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows))) {
        return arrow::Status::Invalid(
            fmt::format("Invalid slice of record batchs: {} out of {}, [chunk info={}]", chunk_info.number_of_rows,
                        rbs_in_file[rbs_idx]->num_rows() - rbs_offset, chunk_info.ToString()));
      }

      auto rb = rbs_in_file[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
      chunk_rb_map[chunk_idx] = rb;
      rbs_offset += chunk_info.number_of_rows;

      assert(rbs_offset <= rbs_in_file[rbs_idx]->num_rows());
      if (rbs_offset == rbs_in_file[rbs_idx]->num_rows()) {
        rbs_idx++;
        rbs_offset = 0;
      }
    }
  }
  return chunk_rb_map;
}

template <typename ReaderT>
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl<ReaderT>::get_chunks(
    const std::vector<int64_t>& chunk_indices, size_t parallelism) {
  assert(opened_);

  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)));

  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  if (unique_chunk_indices.empty()) {
    return std::vector<std::shared_ptr<arrow::RecordBatch>>{};
  }

  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  if (parallelism <= 1) {
    ARROW_ASSIGN_OR_RAISE(chunk_rb_map, read_chunks_from_files(unique_chunk_indices));
  } else {
    auto folly_thread_pool = ThreadPoolHolder::GetThreadPool(parallelism /* parallelism_hint */);
    auto splitted_chunks = split_chunks(unique_chunk_indices, folly_thread_pool->numThreads());
    std::vector<std::future<ChunkRBMapResult>> futures;

    for (const auto& task_indices : splitted_chunks) {
      std::packaged_task<ChunkRBMapResult()> task(
          [this, task_indices]() { return read_chunks_from_files(task_indices); });
      futures.emplace_back(task.get_future());
      folly_thread_pool->add(std::move(task));
    }

    std::vector<ChunkRBMapResult> all_results;
    all_results.reserve(futures.size());
    for (auto& future : futures) {
      all_results.emplace_back(future.get());
    }
    for (auto& result : all_results) {
      ARROW_ASSIGN_OR_RAISE(auto res, std::move(result));
      for (const auto& [chunk_index, rb] : res) {
        chunk_rb_map.emplace(chunk_index, rb);
      }
    }
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  result.reserve(chunk_indices.size());
  for (const auto& chunk_idx : chunk_indices) {
    assert(chunk_rb_map.find(chunk_idx) != chunk_rb_map.end());
    result.emplace_back(chunk_rb_map[chunk_idx]);
  }

  return result;
}

template <typename ReaderT>
arrow::Result<uint64_t> ColumnGroupReaderImpl<ReaderT>::get_chunk_estimated_size(int64_t chunk_index) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  const auto& chunk_info = chunk_infos_[chunk_index];
  if (!chunk_info.memory_size_available) {
    return arrow::Status::NotImplemented("Chunk memory size estimate is not available");
  }
  if (!chunk_info.column_memory_sizes) {
    return chunk_info.avg_memory_size;
  }

  // Files in one column group may have evolved physical schemas and can retain
  // columns removed from the logical group. Sum the logical column estimates
  // so physical-only columns are excluded while the result remains independent
  // of the active projection.
  uint64_t total_size = 0;
  std::unordered_set<std::string_view> seen_columns;
  seen_columns.reserve(column_group_->columns.size());
  for (size_t col_idx = 0; col_idx < column_group_->columns.size(); ++col_idx) {
    const auto& column_name = column_group_->columns[col_idx];
    if (!seen_columns.emplace(column_name).second) {
      return arrow::Status::Invalid("Duplicate column in column group: ", column_name);
    }
    ARROW_ASSIGN_OR_RAISE(auto column_size, get_chunk_column_estimated_size(chunk_index, static_cast<int>(col_idx)));
    if (column_size > std::numeric_limits<uint64_t>::max() - total_size) {
      return arrow::Status::Invalid("Chunk column memory sizes exceed the uint64_t range");
    }
    total_size += column_size;
  }
  return total_size;
}

template <typename ReaderT>
arrow::Result<uint64_t> ColumnGroupReaderImpl<ReaderT>::get_chunk_column_estimated_size(int64_t chunk_index,
                                                                                        int col_idx) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }

  const auto& chunk_info = chunk_infos_[chunk_index];
  const auto& column_memory_sizes = chunk_info.column_memory_sizes;
  if (!chunk_info.memory_size_available) {
    return arrow::Status::NotImplemented("Column memory size estimate is not available");
  }
  if (!column_memory_sizes) {
    return arrow::Status::NotImplemented("Column memory size metadata is not available for this format");
  }
  if (UNLIKELY(col_idx < 0 || static_cast<size_t>(col_idx) >= column_group_->columns.size())) {
    return arrow::Status::Invalid(
        fmt::format("Column index out of range: {} out of {}", col_idx, column_group_->columns.size()));
  }
  const auto it = column_memory_sizes->find(column_group_->columns[col_idx]);
  return it == column_memory_sizes->end() ? uint64_t{0} : it->second;
}

template <typename ReaderT>
arrow::Result<uint64_t> ColumnGroupReaderImpl<ReaderT>::get_chunk_rows(int64_t chunk_index) {
  assert(opened_);
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid(
        fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].number_of_rows;
}

template <typename ReaderT>
const ChunkInfo& ColumnGroupReaderImpl<ReaderT>::get_chunk_info(int64_t chunk_index) const {
  assert(chunk_index >= 0 && static_cast<size_t>(chunk_index) < chunk_infos_.size());
  return chunk_infos_[chunk_index];
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>>
ColumnGroupReaderImpl<ReaderT>::get_chunks_async(const ChunkTask& task) {
  FIU_RETURN_ON(FIUKEY_COLUMN_GROUP_READ_FAIL,
                folly::makeSemiFuture(arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_COLUMN_GROUP_READ_FAIL)))));

  std::vector<ChunkInfo> chunk_infos;
  chunk_infos.reserve(task.chunk_indices.size());
  // Validate the planner contract before opening a backend reader, and copy the
  // metadata needed by continuations so they do not depend on mutable state.
  for (auto chunk_index : task.chunk_indices) {
    if (UNLIKELY(chunk_index < 0 || static_cast<size_t>(chunk_index) >= chunk_infos_.size())) {
      return folly::makeSemiFuture(
          arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(arrow::Status::Invalid(
              fmt::format("Chunk index out of range: {} out of {}", chunk_index, chunk_infos_.size()))));
    }
    const auto& chunk_info = chunk_infos_[chunk_index];
    if (UNLIKELY(chunk_info.file_index != task.file_index)) {
      return folly::makeSemiFuture(
          arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>(arrow::Status::Invalid(fmt::format(
              "Chunk {} belongs to file {}, not task file {}", chunk_index, chunk_info.file_index, task.file_index))));
    }
    chunk_infos.emplace_back(chunk_info);
  }

  // Each task opens independent mutable format-reader state; immutable cached
  // metadata may still be shared across those readers.
  return open_reader_for_file_async(task.file_index)
      .deferValue([range_start = task.range_start, range_end = task.range_end, chunk_infos = std::move(chunk_infos)](
                      arrow::Result<std::shared_ptr<ReaderT>>&& reader_result) mutable
                  -> folly::SemiFuture<arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>> {
        FOLLY_ARROW_ASSIGN_OR_RAISE(auto reader, std::move(reader_result));
        return reader->read_with_range_async(range_start, range_end)
            .deferValue([reader = std::move(reader), chunk_infos = std::move(chunk_infos)](auto&& rb_reader_result)
                            -> arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> {
              // Lifetime-only capture: drain the Arrow reader before releasing
              // the independent FormatReader that produced it.
              (void)reader;
              ARROW_ASSIGN_OR_RAISE(auto rb_reader, std::move(rb_reader_result));
              ARROW_ASSIGN_OR_RAISE(auto rbs, rb_reader->ToRecordBatches());

              // A format may coalesce the range into different batch boundaries;
              // slice it back into one result per logical chunk.
              std::vector<std::shared_ptr<arrow::RecordBatch>> result;
              result.reserve(chunk_infos.size());
              size_t rbs_idx = 0;
              size_t rbs_offset = 0;
              for (const auto& chunk_info : chunk_infos) {
                if (UNLIKELY(rbs_idx >= rbs.size() ||
                             (rbs[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows)) {
                  return arrow::Status::Invalid(fmt::format(
                      "Invalid slice of record batches in async read: [chunk_info={}]", chunk_info.ToString()));
                }
                auto rb = rbs[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
                result.push_back(std::move(rb));
                rbs_offset += chunk_info.number_of_rows;
                if (rbs_offset == rbs[rbs_idx]->num_rows()) {
                  rbs_idx++;
                  rbs_offset = 0;
                }
              }
              return result;
            });
      });
}

template <typename ReaderT>
std::shared_ptr<arrow::Schema> ColumnGroupReaderImpl<ReaderT>::get_schema() const {
  return file_schema_;
}

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    const std::string& predicate,
    const milvus_storage::MetadataCache& cache) {
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }
  const bool cache_enabled =
      cache.enabled() && GetValueNoError<bool>(properties, PROPERTY_READER_METADATA_CACHE_ENABLE);

  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
    }
  }

  if (schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& col_name : filtered_columns) {
      auto field = schema->GetFieldByName(col_name);
      if (!field) {
        return arrow::Status::Invalid(
            "ColumnGroupReader: column '" + col_name +
            "' found in column_group but not in schema. Schema fields: " + schema->ToString());
      }
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }

  auto create_reader = [&](const milvus_storage::MetadataCache& metadata_cache) {
    return metadata_cache.dispatch(
        column_group->format, [&](auto typed_cache) -> arrow::Result<std::unique_ptr<ColumnGroupReader>> {
          if (!typed_cache) {
            return arrow::Status::Invalid("Format reader metadata cache is null");
          }

          using TypedCache = typename decltype(typed_cache)::element_type;
          using ReaderT = typename TypedCache::ReaderType;
          std::unique_ptr<ColumnGroupReader> reader = std::make_unique<ColumnGroupReaderImpl<ReaderT>>(
              out_schema, column_group, properties, filtered_columns, key_retriever, metadata_cache, predicate);
          ARROW_RETURN_NOT_OK(reader->open());
          return reader;
        });
  };

  if (!cache_enabled) {
    return create_reader(milvus_storage::MetadataCache(false));
  }

  return create_reader(cache);
}

folly::SemiFuture<arrow::Result<std::unique_ptr<ColumnGroupReader>>> ColumnGroupReader::create_async(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever,
    const std::string& predicate,
    const milvus_storage::MetadataCache& cache) {
  using ResultType = arrow::Result<std::unique_ptr<ColumnGroupReader>>;
  if (!column_group) {
    return folly::makeSemiFuture(ResultType(arrow::Status::Invalid("Column group cannot be null")));
  }
  const bool cache_enabled =
      cache.enabled() && GetValueNoError<bool>(properties, PROPERTY_READER_METADATA_CACHE_ENABLE);

  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
    }
  }

  if (schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& col_name : filtered_columns) {
      auto field = schema->GetFieldByName(col_name);
      if (!field) {
        return folly::makeSemiFuture(ResultType(
            arrow::Status::Invalid("ColumnGroupReader: column '" + col_name +
                                   "' found in column_group but not in schema. Schema fields: " + schema->ToString())));
      }
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }

  auto create_reader = [&](const milvus_storage::MetadataCache& metadata_cache) {
    return metadata_cache.dispatch(
        column_group->format,
        [&](auto typed_cache) -> folly::SemiFuture<arrow::Result<std::unique_ptr<ColumnGroupReader>>> {
          if (!typed_cache) {
            return folly::makeSemiFuture(ResultType(arrow::Status::Invalid("Format reader metadata cache is null")));
          }

          using TypedCache = typename decltype(typed_cache)::element_type;
          using ReaderT = typename TypedCache::ReaderType;
          auto reader = std::make_unique<ColumnGroupReaderImpl<ReaderT>>(
              out_schema, column_group, properties, filtered_columns, key_retriever, metadata_cache, predicate);
          auto* reader_ptr = reader.get();
          // The continuation owns the unique_ptr while open_async() uses reader_ptr.
          return reader_ptr->open_async().deferValue([reader = std::move(reader)](arrow::Status status) mutable
                                                     -> arrow::Result<std::unique_ptr<ColumnGroupReader>> {
            ARROW_RETURN_NOT_OK(status);
            return std::move(reader);
          });
        });
  };

  if (!cache_enabled) {
    return create_reader(milvus_storage::MetadataCache(false));
  }

  return create_reader(cache);
}

}  // namespace milvus_storage::api
