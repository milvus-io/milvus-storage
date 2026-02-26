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

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <arrow/compute/api.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY
#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage::api {

class ColumnGroupLazyReaderImpl : public ColumnGroupLazyReader {
  public:
  ColumnGroupLazyReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                            const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                            const milvus_storage::api::Properties& properties,
                            const std::vector<std::string>& needed_columns,
                            const std::function<std::string(const std::string&)>& key_retriever);

  ~ColumnGroupLazyReaderImpl() = default;

  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices,
                                                    size_t parallelism = 1) override;

  private:
  arrow::Status prepare_format_readers(const std::vector<int64_t>& row_indices);
  arrow::Result<std::shared_ptr<arrow::Table>> take_rows_from_files(const std::vector<int64_t>& row_indices);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  std::mutex prepare_mutex_;
  std::vector<std::shared_ptr<FormatReader>> loaded_format_readers_;
};

ColumnGroupLazyReaderImpl::ColumnGroupLazyReaderImpl(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever),
      loaded_format_readers_(column_group->files.size()) {}

static inline arrow::Result<std::pair<uint32_t, int64_t>> get_index_and_offset_of_file(
    const std::vector<ColumnGroupFile>& files, const int64_t& global_row_index) {
  int64_t row_index_remain = global_row_index;

  for (uint32_t i = 0; i < files.size(); i++) {
    if (files[i].start_index < 0 || files[i].end_index < 0) {
      return arrow::Status::Invalid(
          fmt::format("Invalid start/end index in [file_index={}, path={}]", i, files[i].path));
    }

    int64_t num_of_rows_in_file = (files[i].end_index - files[i].start_index);
    if (row_index_remain < num_of_rows_in_file) {
      // use the physical row index in file
      return std::make_pair(i, row_index_remain + files[i].start_index);
    }

    row_index_remain -= num_of_rows_in_file;
  }

  return arrow::Status::Invalid(
      fmt::format("Row index is greater than the maximum range, [row_index={}]", global_row_index));
}

#if 0
static arrow::Result<std::shared_ptr<arrow::Table>> reorder_tables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables, const std::vector<int64_t>& indices) {
  ARROW_ASSIGN_OR_RAISE(auto concatenated_table, arrow::ConcatenateTables(tables));
  arrow::compute::ExecContext context;
  arrow::compute::TakeOptions options = arrow::compute::TakeOptions::Defaults();

  auto indices_array = std::make_shared<arrow::Int64Array>(indices.size(), arrow::Buffer::Wrap(indices));

  ARROW_ASSIGN_OR_RAISE(arrow::Datum result_datum,
                        arrow::compute::Take(concatenated_table, indices_array, options, &context));

  return result_datum.table();
}
#endif

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

arrow::Status ColumnGroupLazyReaderImpl::prepare_format_readers(const std::vector<int64_t>& row_indices) {
  std::lock_guard<std::mutex> lock(prepare_mutex_);
  const auto& cg_files = column_group_->files;
  for (const auto& row_index : row_indices) {
    uint32_t file_index;
    [[maybe_unused]] int64_t _unused_row_index_in_file;

    if (row_index < 0) {
      return arrow::Status::Invalid(fmt::format("Row index is less than 0, [row_index={}]", row_index));
    }
    ARROW_ASSIGN_OR_RAISE(std::tie(file_index, _unused_row_index_in_file),
                          get_index_and_offset_of_file(cg_files, row_index));
    if (!loaded_format_readers_[file_index]) {
      ARROW_ASSIGN_OR_RAISE(loaded_format_readers_[file_index],
                            FormatReader::create(schema_, column_group_->format, cg_files[file_index], properties_,
                                                 needed_columns_, key_retriever_));
    }
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl::take_rows_from_files(
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

    ARROW_ASSIGN_OR_RAISE(auto cloned_reader, loaded_format_readers_[file_index]->clone_reader());
    ARROW_ASSIGN_OR_RAISE(auto table, cloned_reader->take(indices_in_files[file_index]));
    tables.emplace_back(table);
  }

  // won't copy table with same schema
  return arrow::ConcatenateTables(tables);
}

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl::take(const std::vector<int64_t>& row_indices,
                                                                             size_t parallelism) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)));

  for (int i = 1; i < row_indices.size(); i++) {
    if (row_indices[i] <= row_indices[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Input row indices is not sorted or not unique,[index={}, row_index={}]", i, row_indices[i]));
    }
  }

  ARROW_RETURN_NOT_OK(prepare_format_readers(row_indices));

  if (parallelism <= 1) {
    return take_rows_from_files(row_indices);
  }

  auto folly_thread_pool = ThreadPoolHolder::GetThreadPool(parallelism /* parallelism_hint */);
  auto splitted_row_indices = split_row_indices(row_indices, folly_thread_pool->numThreads());
  std::vector<std::shared_ptr<arrow::Table>> result_tables;
  std::vector<std::future<arrow::Result<std::shared_ptr<arrow::Table>>>> futures;

  for (const auto& task_row_indices : splitted_row_indices) {
    std::packaged_task<arrow::Result<std::shared_ptr<arrow::Table>>()> task(
        [this, task_row_indices]() { return take_rows_from_files(task_row_indices); });
    futures.emplace_back(task.get_future());
    folly_thread_pool->add(std::move(task));
  }

  // Wait for all futures to complete before checking errors,
  // to avoid early return while tasks still hold `this`.
  std::vector<arrow::Result<std::shared_ptr<arrow::Table>>> all_results;
  for (auto& future : futures) {
    all_results.emplace_back(future.get());
  }
  for (auto& result : all_results) {
    ARROW_ASSIGN_OR_RAISE(auto table, std::move(result));
    result_tables.emplace_back(std::move(table));
  }

  // won't copy table with same schema
  return arrow::ConcatenateTables(result_tables);

#if 0
  // support reorder in follow logical
  // no need reorder
  if (!need_reorder) {
    // table with same schema won't copy
    tables.erase(std::remove(tables.begin(), tables.end(), nullptr), tables.end());
    return arrow::ConcatenateTables(tables);
  } else {
    std::vector<int64_t> row_offset_in_tables(tables.size());
    std::vector<int64_t> reorder_indices(row_indices.size());

    int64_t current_offset = 0;
    for (size_t i = 0; i < tables.size(); ++i) {
      if (!tables[i]) {
        row_offset_in_tables[i] = -1;
        continue;
      }
      row_offset_in_tables[i] = current_offset;
      current_offset += tables[i]->num_rows();
    }

    for (int i = 0; i < row_indices.size(); ++i) {
      auto rbidx = row_index_to_rbidx[row_indices[i]];

      assert(rbidx.first < row_offset_in_tables.size() && row_offset_in_tables[rbidx.first] != -1);
      reorder_indices[i] = row_offset_in_tables[rbidx.first] + rbidx.second;
    }
    tables.erase(std::remove(tables.begin(), tables.end(), nullptr), tables.end());
    return reorder_tables(tables, reorder_indices);
  }

  assert(false);
  return arrow::Status::Invalid("Unreachable code");
#endif
}

arrow::Result<std::unique_ptr<ColumnGroupLazyReader>> ColumnGroupLazyReader::create(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
    const milvus_storage::api::Properties& properties,
    const std::vector<std::string>& needed_columns,
    const std::function<std::string(const std::string&)>& key_retriever) {
  std::shared_ptr<arrow::Schema> out_schema;
  std::vector<std::string> filtered_columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& col_name : needed_columns) {
    if (std::find(column_group->columns.begin(), column_group->columns.end(), col_name) !=
        column_group->columns.end()) {
      filtered_columns.emplace_back(col_name);
      auto field = schema->GetFieldByName(col_name);
      assert(field);
      fields.emplace_back(field);
    }
  }

  out_schema = std::make_shared<arrow::Schema>(fields);

  return std::make_unique<ColumnGroupLazyReaderImpl>(out_schema, column_group, properties, filtered_columns,
                                                     key_retriever);
}

};  // namespace milvus_storage::api
