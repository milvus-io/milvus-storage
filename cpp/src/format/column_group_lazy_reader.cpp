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
#include <span>
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
#include <folly/ScopeGuard.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/format/iceberg/iceberg_format_reader.h"
#include "milvus-storage/format/lance/lance_table_reader.h"
#include "milvus-storage/format/paimon/paimon_format_reader.h"
#include "milvus-storage/format/parquet/parquet_format_reader.h"
#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage::api {

template <typename ReaderT>
class ColumnGroupLazyReaderImpl : public ColumnGroupLazyReader {
  public:
  ColumnGroupLazyReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                            const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                            const std::shared_ptr<const milvus_storage::api::Properties>& properties,
                            const std::vector<std::string>& needed_columns,
                            const std::function<std::string(const std::string&)>& key_retriever,
                            const milvus_storage::MetadataCache& cache);

  ~ColumnGroupLazyReaderImpl() override = default;

  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices,
                                                    size_t parallelism = 1) override;
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(TakeTask&& task) override;

  private:
  arrow::Result<std::shared_ptr<arrow::Table>> take_files(std::span<const TakeTask> tasks);
  arrow::Result<std::shared_ptr<ReaderT>> open_reader_for_file(size_t file_index);
  folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>> open_reader_for_file_async(size_t file_index);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  std::shared_ptr<const milvus_storage::api::Properties> properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;
  milvus_storage::MetadataCache cache_;
};

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
                        -> folly::SemiFuture<arrow::Result<std::shared_ptr<ReaderT>>> {
          FOLLY_ARROW_ASSIGN_OR_RAISE(auto metadata, std::move(metadata_result));
          return ReaderT::MetaTrait::create_from_metadata_async(std::move(metadata), file, read_schema, needed_columns,
                                                                "");
        });
  }

  // Formats without a complete async metadata path may block here before
  // returning their ready future.
  return folly::makeSemiFuture(open_reader_for_file(file_index));
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl<ReaderT>::take_files(
    std::span<const TakeTask> tasks) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (const auto& task : tasks) {
    ARROW_ASSIGN_OR_RAISE(auto reader, open_reader_for_file(task.file_index));
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(task.file_offsets));
    tables.emplace_back(std::move(table));
  }
  if (tables.size() == 1) {
    return tables.front();
  }
  return arrow::ConcatenateTables(tables);
}

template <typename ReaderT>
arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl<ReaderT>::take(
    const std::vector<int64_t>& row_indices, size_t parallelism) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)));

  auto pool = parallelism > 1 ? ThreadPoolHolder::GetThreadPool(parallelism) : nullptr;
  const size_t workers = pool ? pool->numThreads() : 1;
  const size_t rows_per_batch =
      workers > 1 && row_indices.size() >= workers ? 1 + (row_indices.size() - 1) / workers : 0;
  ARROW_ASSIGN_OR_RAISE(auto tasks, TakeTask::Build({column_group_}, row_indices, rows_per_batch));
  if (rows_per_batch == 0) {
    return take_files(tasks);
  }

  std::vector<std::future<arrow::Result<std::shared_ptr<arrow::Table>>>> futures;
  // Workers borrow the plan and reader. Drain outstanding work even if a read fails.
  auto drain = folly::makeGuard([&]() noexcept {
    for (auto& future : futures) {
      if (future.valid()) {
        future.wait();
      }
    }
  });
  for (size_t first = 0; first < tasks.size();) {
    const size_t batch_number = tasks[first].original_position / rows_per_batch;
    size_t end = first + 1;
    while (end < tasks.size() && tasks[end].original_position / rows_per_batch == batch_number) {
      ++end;
    }
    auto batch = std::span<const TakeTask>(tasks).subspan(first, end - first);
    first = end;
    std::packaged_task<arrow::Result<std::shared_ptr<arrow::Table>>()> work(
        [this, batch]() { return take_files(batch); });
    futures.emplace_back(work.get_future());
    pool->add(std::move(work));
  }

  std::vector<std::shared_ptr<arrow::Table>> result_tables;
  for (auto& future : futures) {
    ARROW_ASSIGN_OR_RAISE(auto table, future.get());
    result_tables.emplace_back(std::move(table));
  }
  return arrow::ConcatenateTables(result_tables);
}

template <typename ReaderT>
folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> ColumnGroupLazyReaderImpl<ReaderT>::take_async(
    TakeTask&& task) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)))));

  // Open one independent reader for this file-scoped task; no mutable format
  // reader is shared with another in-flight take.
  return open_reader_for_file_async(task.file_index)
      .deferValue([rows_in_file = std::move(task.file_offsets)](arrow::Result<std::shared_ptr<ReaderT>>&& reader_result)
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
    const std::shared_ptr<const milvus_storage::api::Properties>& properties,
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
    const std::shared_ptr<const milvus_storage::api::Properties>& properties,
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
