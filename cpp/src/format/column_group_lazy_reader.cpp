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

#include "milvus-storage/format/column_group_lazy_reader.h"

#include <algorithm>
#include <future>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <fmt/format.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
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

template <typename ReaderT>
class ColumnGroupLazyReaderImpl : public ColumnGroupLazyReader {
  public:
  ColumnGroupLazyReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                            const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                            const milvus_storage::api::Properties& properties,
                            const std::vector<std::string>& needed_columns,
                            const std::function<std::string(const std::string&)>& key_retriever,
                            const milvus_storage::MetadataCache& cache);

  ~ColumnGroupLazyReaderImpl() override = default;

  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices,
                                                    size_t parallelism = 1) override;
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(const TakeTask& task) override;

  private:
  arrow::Status validate_row_indices(const std::vector<int64_t>& row_indices) const;
  arrow::Result<std::shared_ptr<arrow::Table>> take_rows_from_files(const std::vector<int64_t>& row_indices);
  arrow::Result<std::shared_ptr<ReaderT>> open_reader_for_file(size_t file_index);
  folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>> open_reader_for_file_async(size_t file_index);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  milvus_storage::MetadataCache cache_;
};

// FIXME(jiaqizho): This single-row lookup rescans files from the beginning. Batch callers
// with sorted row indices could carry a forward file cursor and reduce the
// overall lookup cost from O(rows * files) to O(rows + files).
static inline arrow::Result<std::pair<uint32_t, int64_t>> get_index_and_offset_of_file(
    const std::vector<ColumnGroupFile>& files, const int64_t& global_row_index) {
  int64_t row_index_remain = global_row_index;

  for (uint32_t i = 0; i < files.size(); i++) {
    if (files[i].start_index < 0 || files[i].end_index < 0) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", i, files[i].path));
    }

    int64_t num_of_rows_in_file = files[i].end_index - files[i].start_index;
    if (row_index_remain < num_of_rows_in_file) {
      return std::make_pair(i, row_index_remain + files[i].start_index);
    }

    row_index_remain -= num_of_rows_in_file;
  }

  return arrow::Status::Invalid(
      fmt::format("Row index is greater than the maximum range, [row_index={}]", global_row_index));
}

static arrow::Status validate_sorted_unique_row_indices(const std::vector<int64_t>& row_indices) {
  for (size_t i = 1; i < row_indices.size(); i++) {
    if (row_indices[i] <= row_indices[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Input row indices is not sorted or not unique,[index={}, row_index={}]", i, row_indices[i]));
    }
  }
  return arrow::Status::OK();
}

static std::vector<std::vector<int64_t>> split_row_indices(const std::vector<int64_t>& row_indices,
                                                           uint64_t parallel_degree) {
  if (parallel_degree == 0 || row_indices.size() < parallel_degree) {
    return std::vector<std::vector<int64_t>>{row_indices};
  }

  uint64_t avg_rows = row_indices.size() / parallel_degree;
  std::vector<std::vector<int64_t>> splitted_row_indices(parallel_degree);
  for (uint64_t i = 0; i < parallel_degree; i++) {
    uint64_t start = i * avg_rows;
    uint64_t end = (i + 1) * avg_rows;
    if (i == parallel_degree - 1) {
      end = row_indices.size();
    }
    splitted_row_indices[i] = std::vector<int64_t>(row_indices.begin() + start, row_indices.begin() + end);
  }

  return splitted_row_indices;
}

template <typename ReaderT>
arrow::Status ColumnGroupLazyReaderImpl<ReaderT>::validate_row_indices(const std::vector<int64_t>& row_indices) const {
  const auto& cg_files = column_group_->files;
  for (const auto& row_index : row_indices) {
    uint32_t file_index;
    [[maybe_unused]] int64_t _unused_row_index_in_file;

    if (row_index < 0) {
      return arrow::Status::Invalid(fmt::format("Row index is less than 0, [row_index={}]", row_index));
    }
    ARROW_ASSIGN_OR_RAISE(std::tie(file_index, _unused_row_index_in_file),
                          get_index_and_offset_of_file(cg_files, row_index));
  }
  return arrow::Status::OK();
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<ReaderT>> ColumnGroupLazyReaderImpl<ReaderT>::open_reader_for_file(size_t file_index) {
  if (file_index >= column_group_->files.size()) {
    return arrow::Status::Invalid("Column group file index out of range: ", file_index,
                                  " >= ", column_group_->files.size());
  }

  auto file = column_group_->files[file_index];
  if (!cache_.enabled()) {
    ARROW_ASSIGN_OR_RAISE(auto reader, FormatReader::create(schema_, column_group_->format, file, properties_,
                                                            needed_columns_, key_retriever_));
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
    return FormatReader::create_from_metadata<ReaderT>(metadata, file, schema_, needed_columns_, "");
  }

  return arrow::Status::Invalid("Unreachable code");
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>>
ColumnGroupLazyReaderImpl<ReaderT>::open_reader_for_file_async(size_t file_index) {
  if (file_index >= column_group_->files.size()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<ReaderT>>(arrow::Status::Invalid(
        "Column group file index out of range: ", file_index, " >= ", column_group_->files.size())));
  }

  auto file = column_group_->files[file_index];
  if (!cache_.enabled()) {
    // No cache: create a fresh reader and let the format decide whether open is native async.
    return FormatReader::create_async(schema_, column_group_->format, file, properties_, needed_columns_,
                                      key_retriever_)
        .deferValue([format = column_group_->format](arrow::Result<std::shared_ptr<FormatReader>>&& reader_result)
                        -> arrow::Result<std::shared_ptr<ReaderT>> {
          ARROW_ASSIGN_OR_RAISE(auto reader, std::move(reader_result));
          auto typed_reader = std::dynamic_pointer_cast<ReaderT>(reader);
          if (!typed_reader) {
            return arrow::Status::Invalid("FormatReader::create_async returned incompatible reader for format: ",
                                          format);
          }
          return typed_reader;
        });
  }

  if constexpr (FormatReaderWithAsyncMetadata<ReaderT>) {
    // Cache immutable file metadata only; every take task still receives its
    // own reader with task-specific projection state.
    auto typed_cache = cache_.get<ReaderT>();
    auto key = ReaderT::MetaTrait::cache_key(file);
    return typed_cache
        ->get_or_open_async(key,
                            [file, properties = properties_, key_retriever = key_retriever_]() {
                              return ReaderT::MetaTrait::load_metadata_async(file, properties, key_retriever);
                            })
        .deferValue([file, read_schema = schema_, needed_columns = needed_columns_](
                        arrow::Result<typename ReaderT::MetaTrait::MetadataPtr>&& metadata_result)
                        -> arrow::Result<std::shared_ptr<ReaderT>> {
          ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(metadata_result));
          return FormatReader::create_from_metadata<ReaderT>(std::move(metadata), file, read_schema, needed_columns,
                                                             "");
        });
  }

  // Synchronous metadata formats may block here before returning their ready future.
  return folly::makeSemiFuture(open_reader_for_file(file_index));
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl<ReaderT>::take_rows_from_files(
    const std::vector<int64_t>& row_indices) {
  const auto& cg_files = column_group_->files;
  std::vector<std::vector<int64_t>> indices_in_files(cg_files.size());
  for (const auto& row_index : row_indices) {
    uint32_t file_index;
    int64_t row_index_in_file;
    ARROW_ASSIGN_OR_RAISE(std::tie(file_index, row_index_in_file), get_index_and_offset_of_file(cg_files, row_index));
    indices_in_files[file_index].emplace_back(row_index_in_file);
  }

  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (size_t file_index = 0; file_index < indices_in_files.size(); file_index++) {
    if (indices_in_files[file_index].empty()) {
      continue;
    }

    ARROW_ASSIGN_OR_RAISE(auto reader, open_reader_for_file(file_index));
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices_in_files[file_index]));
    tables.emplace_back(table);
  }

  // won't copy table with same schema
  return arrow::ConcatenateTables(tables);
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl<ReaderT>::take(
    const std::vector<int64_t>& row_indices, size_t parallelism) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)));

  ARROW_RETURN_NOT_OK(validate_sorted_unique_row_indices(row_indices));
  ARROW_RETURN_NOT_OK(validate_row_indices(row_indices));

  if (parallelism <= 1) {
    return take_rows_from_files(row_indices);
  }

  auto folly_thread_pool = ThreadPoolHolder::GetThreadPool(parallelism /* parallelism_hint */);
  auto splitted_row_indices = split_row_indices(row_indices, folly_thread_pool->numThreads());
  std::vector<std::future<arrow::Result<std::shared_ptr<arrow::Table>>>> futures;

  for (const auto& task_row_indices : splitted_row_indices) {
    std::packaged_task<arrow::Result<std::shared_ptr<arrow::Table>>()> task(
        [this, task_row_indices]() { return take_rows_from_files(task_row_indices); });
    futures.emplace_back(task.get_future());
    folly_thread_pool->add(std::move(task));
  }

  std::vector<std::shared_ptr<arrow::Table>> result_tables;
  result_tables.reserve(futures.size());
  std::vector<arrow::Result<std::shared_ptr<arrow::Table>>> all_results;
  all_results.reserve(futures.size());
  for (auto& future : futures) {
    all_results.emplace_back(future.get());
  }
  for (auto& result : all_results) {
    ARROW_ASSIGN_OR_RAISE(auto table, std::move(result));
    result_tables.emplace_back(std::move(table));
  }

  return arrow::ConcatenateTables(result_tables);
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> ColumnGroupLazyReaderImpl<ReaderT>::take_async(
    const TakeTask& task) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)))));

  const auto& cg_files = column_group_->files;
  std::vector<int64_t> rows_in_file;
  rows_in_file.reserve(task.row_indices.size());
  // Planning keeps global row indices; the format reader consumes file-local offsets.
  for (auto global_row : task.row_indices) {
    FOLLY_ARROW_ASSIGN_OR_RAISE(auto file_and_offset, get_index_and_offset_of_file(cg_files, global_row));
    if (file_and_offset.first != task.file_index) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(arrow::Status::Invalid(
          fmt::format("TakeTask row does not belong to task file. [row={}, expected_file={}, actual_file={}]",
                      global_row, task.file_index, file_and_offset.first))));
    }
    rows_in_file.push_back(file_and_offset.second);
  }

  // Open one independent reader for this file-scoped task; no mutable format
  // reader is shared with another in-flight take.
  return open_reader_for_file_async(task.file_index)
      .deferValue([rows_in_file = std::move(rows_in_file)](arrow::Result<std::shared_ptr<ReaderT>>&& reader_result)
                      -> folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> {
        FOLLY_ARROW_ASSIGN_OR_RAISE(auto reader, std::move(reader_result));
        return reader->take_async(rows_in_file)
            .deferValue(
                [reader = std::move(reader)](auto&& table_result) -> arrow::Result<std::shared_ptr<arrow::Table>> {
                  // Lifetime-only capture: backend state must outlive the async take.
                  (void)reader;
                  return std::move(table_result);
                });
      });
}

template <typename ReaderT>
ColumnGroupLazyReaderImpl<ReaderT>::ColumnGroupLazyReaderImpl(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
    const milvus_storage::MetadataCache& cache)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      cache_(cache) {}

arrow::Result<std::unique_ptr<ColumnGroupLazyReader>> ColumnGroupLazyReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever,
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
      assert(field);
      fields.emplace_back(field);
    }
    out_schema = std::make_shared<arrow::Schema>(fields);
  }
  // When schema is nullptr, out_schema stays nullptr;
  // the RecordBatches returned by the format reader will carry the file schema.

  auto create_reader = [&](const milvus_storage::MetadataCache& metadata_cache) {
    return metadata_cache.dispatch(
        column_group->format, [&](auto typed_cache) -> arrow::Result<std::unique_ptr<ColumnGroupLazyReader>> {
          if (!typed_cache) {
            return arrow::Status::Invalid("Format reader metadata cache is null");
          }

          using TypedCache = typename decltype(typed_cache)::element_type;
          using ReaderT = typename TypedCache::ReaderType;
          return std::make_unique<ColumnGroupLazyReaderImpl<ReaderT>>(out_schema, column_group, properties,
                                                                      filtered_columns, key_retriever, metadata_cache);
        });
  };

  if (!cache_enabled) {
    return create_reader(milvus_storage::MetadataCache(false));
  }

  return create_reader(cache);
}

}  // namespace milvus_storage::api
