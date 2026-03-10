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

#include <map>
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

  arrow::Result<std::shared_ptr<arrow::Table>> take(const std::vector<int64_t>& row_indices) override;

  arrow::Result<std::vector<TakeTask>> get_natural_tasks(const std::vector<int64_t>& row_indices) override;
  folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> take_async(const TakeTask& task) override;

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

arrow::Result<std::shared_ptr<arrow::Table>> ColumnGroupLazyReaderImpl::take(const std::vector<int64_t>& row_indices) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)));

  for (int i = 1; i < row_indices.size(); i++) {
    if (row_indices[i] <= row_indices[i - 1]) {
      return arrow::Status::Invalid(
          fmt::format("Input row indices is not sorted or not unique,[index={}, row_index={}]", i, row_indices[i]));
    }
  }

  ARROW_RETURN_NOT_OK(prepare_format_readers(row_indices));
  return take_rows_from_files(row_indices);
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

arrow::Result<std::vector<TakeTask>> ColumnGroupLazyReaderImpl::get_natural_tasks(
    const std::vector<int64_t>& row_indices) {
  const auto& files = column_group_->files;
  // file_idx -> [(global_row_index, original_position)]
  std::map<uint32_t, std::vector<std::pair<int64_t, size_t>>> file_groups;

  for (size_t pos = 0; pos < row_indices.size(); ++pos) {
    ARROW_ASSIGN_OR_RAISE(auto file_and_offset, get_index_and_offset_of_file(files, row_indices[pos]));
    auto [file_idx, _] = file_and_offset;
    file_groups[file_idx].push_back({row_indices[pos], pos});
  }

  std::vector<TakeTask> tasks;
  for (auto& [file_idx, rows_and_positions] : file_groups) {
    TakeTask task;
    task.file_index = file_idx;
    task.row_indices.reserve(rows_and_positions.size());
    task.original_positions.reserve(rows_and_positions.size());
    for (auto& [row, pos] : rows_and_positions) {
      task.row_indices.push_back(row);
      task.original_positions.push_back(pos);
    }
    tasks.push_back(std::move(task));
  }
  return tasks;
}

folly::SemiFuture<arrow::Result<std::shared_ptr<arrow::Table>>> ColumnGroupLazyReaderImpl::take_async(
    const TakeTask& task) {
  FIU_RETURN_ON(FIUKEY_TAKE_ROWS_FAIL,
                folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(
                    arrow::Status::IOError(fmt::format("Injected fault: {}", FIUKEY_TAKE_ROWS_FAIL)))));

  // prepare must complete before async submission (has mutex, opens metadata)
  auto prepare_status = prepare_format_readers(task.row_indices);
  if (!prepare_status.ok()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(prepare_status));
  }

  // Convert global row indices to file-local indices
  const auto& cg_files = column_group_->files;
  std::vector<int64_t> rows_in_file;
  rows_in_file.reserve(task.row_indices.size());
  for (auto global_row : task.row_indices) {
    auto result = get_index_and_offset_of_file(cg_files, global_row);
    if (!result.ok()) {
      return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(result.status()));
    }
    auto [file_idx, row_in_file] = result.ValueOrDie();
    rows_in_file.push_back(row_in_file);
  }

  auto cloned = loaded_format_readers_[task.file_index]->clone_reader();
  if (!cloned.ok()) {
    return folly::makeSemiFuture(arrow::Result<std::shared_ptr<arrow::Table>>(cloned.status()));
  }
  auto reader = cloned.MoveValueUnsafe();

  // Captures `reader` to extend its lifetime through the async chain.
  auto forward_result = [reader](auto&& table_result) -> arrow::Result<std::shared_ptr<arrow::Table>> {
    return std::move(table_result);
  };

  return reader->take_async(rows_in_file).deferValue(std::move(forward_result));
}

};  // namespace milvus_storage::api
