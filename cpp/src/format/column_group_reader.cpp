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

#include <unordered_map>
#include <vector>
#include <algorithm>

#include <arrow/array/util.h>
#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/common/lrucache.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"  // for UNLIKELY

namespace milvus_storage::api {

using milvus_storage::RowGroupInfo;
struct ChunkInfo {
  public:
  size_t file_index;               // current chunk belong which file
  size_t row_offset_in_row_group;  // the starting row offset of this row group in its file
  size_t row_offset_in_file;       // the starting row offset of file
  size_t number_of_rows;           // number of rows in this row group
  size_t row_group_index_in_file;  // the index of this row group in its file
  size_t global_row_end;           // the ending row offset of this row group in the whole chunk reader
  size_t avg_memory_size;          // average memory usage of this row group

  ChunkInfo() = default;
  std::string ToString() const;
};

class ColumnGroupReaderImpl : public ColumnGroupReader {
  public:
  ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                        const std::shared_ptr<milvus_storage::api::ColumnGroup>& column_group,
                        const milvus_storage::api::Properties& properties,
                        const std::vector<std::string>& needed_columns,
                        const std::function<std::string(const std::string&)>& key_retriever);

  ~ColumnGroupReaderImpl() = default;

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

  protected:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<milvus_storage::api::ColumnGroup> column_group_;
  milvus_storage::api::Properties properties_;
  std::vector<std::string> needed_columns_;
  std::function<std::string(const std::string&)> key_retriever_;

  // will be initialized after call open()
  std::vector<ChunkInfo> chunk_infos_;
  std::vector<std::vector<RowGroupInfo>> row_group_infos_;
  size_t total_rows_;

  std::vector<std::unique_ptr<FormatReader>> format_readers_;
};  // ColumnGroupReaderImpl

arrow::Result<std::unique_ptr<ColumnGroupReader>> ColumnGroupReader::create(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<milvus_storage::api::ColumnGroup> column_group,
    const std::vector<std::string>& needed_columns,
    const milvus_storage::api::Properties& properties,
    const std::function<std::string(const std::string&)>& key_retriever) {
  std::unique_ptr<ColumnGroupReader> reader = nullptr;
  if (!column_group) {
    return arrow::Status::Invalid("Column group cannot be null");
  }

  // Generate the output schema with only the needed columns
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
  reader = std::make_unique<milvus_storage::api::ColumnGroupReaderImpl>(out_schema, column_group, properties,
                                                                        filtered_columns, key_retriever);
  ARROW_RETURN_NOT_OK(reader->open());
  return std::move(reader);
}

std::string ChunkInfo::ToString() const {
  std::stringstream ss;
  ss << "ChunkInfo{"
     << "file_index=" << file_index << ", row_offset_in_row_group=" << row_offset_in_row_group
     << ", row_offset_in_file=" << row_offset_in_file << ", number_of_rows=" << number_of_rows
     << ", row_group_index_in_file=" << row_group_index_in_file << ", global_row_end=" << global_row_end
     << ", avg_memory_size=" << avg_memory_size << "}";
  return ss.str();
}

ColumnGroupReaderImpl::ColumnGroupReaderImpl(const std::shared_ptr<arrow::Schema>& schema,
                                             const std::shared_ptr<api::ColumnGroup>& column_group,
                                             const api::Properties& properties,
                                             const std::vector<std::string>& needed_columns,
                                             const std::function<std::string(const std::string&)>& key_retriever)
    : schema_(schema),
      column_group_(column_group),
      properties_(properties),
      needed_columns_(needed_columns),
      key_retriever_(key_retriever) {}

arrow::Status ColumnGroupReaderImpl::open() {
  const auto& cg_files = column_group_->files;

  // init chunk infos
  size_t rows_in_all_files = 0;
  for (size_t file_idx = 0; file_idx < cg_files.size(); ++file_idx) {
    auto& cg_file = cg_files[file_idx];

    ARROW_ASSIGN_OR_RAISE(auto format_reader, FormatReader::create(schema_, column_group_->format, cg_file.path,
                                                                   properties_, needed_columns_, key_retriever_));
    ARROW_RETURN_NOT_OK(format_reader->open());
    ARROW_ASSIGN_OR_RAISE(auto row_group_in_file, format_reader->get_row_group_infos());

    size_t rows_in_file = 0;
    if (cg_file.start_index.has_value() && cg_file.end_index.has_value()) {
      const auto& start_index = cg_file.start_index.value();
      const auto& end_index = cg_file.end_index.value();

      assert(start_index >= 0 && end_index > 0 && start_index < end_index);

      for (size_t j = 0; j < row_group_in_file.size(); ++j) {
        size_t rg_start = row_group_in_file[j].start_offset;
        size_t rg_end = row_group_in_file[j].end_offset;

        // calculate the overlap range
        size_t overlap_start = std::max((size_t)start_index, rg_start);
        size_t overlap_end = std::min((size_t)end_index, rg_end);

        // if the overlap range is valid, create the chunk info
        if (overlap_start < overlap_end) {
          rows_in_file += (overlap_end - overlap_start);
          chunk_infos_.emplace_back(ChunkInfo{
              .file_index = file_idx,
              .row_offset_in_row_group = overlap_start - rg_start,
              .row_offset_in_file = overlap_start,
              .number_of_rows = overlap_end - overlap_start,
              .row_group_index_in_file = j,
              .global_row_end = rows_in_all_files + rows_in_file,
              .avg_memory_size = row_group_in_file[j].memory_size * (overlap_end - overlap_start) / (rg_end - rg_start),
          });
        }
      }
    } else {
      // create the chunk infos with row group indices
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
        });
      }
    }

    row_group_infos_.emplace_back(row_group_in_file);
    format_readers_.emplace_back(std::move(format_reader));
    rows_in_all_files += rows_in_file;
  }

  total_rows_ = rows_in_all_files;
  return arrow::Status::OK();
}

size_t ColumnGroupReaderImpl::total_number_of_chunks() const {
  assert(!format_readers_.empty());
  return chunk_infos_.size();
}

size_t ColumnGroupReaderImpl::total_rows() const {
  assert(!format_readers_.empty());
  return total_rows_;
}

arrow::Result<std::vector<int64_t>> ColumnGroupReaderImpl::get_chunk_indices(const std::vector<int64_t>& row_indices) {
  assert(!format_readers_.empty());
  std::unordered_set<int64_t> unique_chunk_indices;
  std::vector<int64_t> chunk_indices;
  for (int64_t row_index : row_indices) {
    auto it = std::upper_bound(chunk_infos_.begin(), chunk_infos_.end(), row_index,
                               [](int64_t a, const ChunkInfo& b) { return a < b.global_row_end; });
    auto chunk_index = std::distance(chunk_infos_.begin(), it);
    if (chunk_index >= chunk_infos_.size()) {
      return arrow::Status::Invalid("Row index out of range: " + std::to_string(row_index));
    }

    if (unique_chunk_indices.find(chunk_index) == unique_chunk_indices.end()) {
      unique_chunk_indices.insert(chunk_index);
      chunk_indices.emplace_back(chunk_index);
    }
  }

  return chunk_indices;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl::get_chunk(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (chunk_index < 0 || chunk_index >= chunk_infos_.size()) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  auto chunk_info = chunk_infos_[chunk_index];

  ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[chunk_info.file_index]->get_chunk(chunk_info.row_group_index_in_file));

  if (chunk_info.row_offset_in_row_group != 0 || chunk_info.number_of_rows != rb->num_rows()) {
    rb = rb->Slice(chunk_info.row_offset_in_row_group, chunk_info.number_of_rows);
  }

  return rb;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>> ColumnGroupReaderImpl::get_chunks(
    const std::vector<int64_t>& chunk_indices) {
  assert(!format_readers_.empty());
  std::unordered_map<int64_t, std::shared_ptr<arrow::RecordBatch>> chunk_rb_map;
  std::vector<std::vector<int64_t>> chunk_idxs_in_files(format_readers_.size());

  // remove duplicate chunk indices and sort by chunk index
  std::vector<int64_t> unique_chunk_indices(chunk_indices.begin(), chunk_indices.end());
  std::sort(unique_chunk_indices.begin(), unique_chunk_indices.end());
  unique_chunk_indices.erase(std::unique(unique_chunk_indices.begin(), unique_chunk_indices.end()),
                             unique_chunk_indices.end());

  // 1. Grouping row groups by file
  for (int64_t chunk_index : unique_chunk_indices) {
    if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
      return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                    std::to_string(chunk_infos_.size()));
    }

    const auto& chunk_info = chunk_infos_[chunk_index];

    // must be sorted.
    chunk_idxs_in_files[chunk_info.file_index].emplace_back(chunk_index);
  }

  // 2. Read with range and fill chunk_rb_map
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

    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs_in_file;
    for (auto& range : ranges_in_file) {
      ARROW_ASSIGN_OR_RAISE(auto rbreader, format_readers_[file_idx]->read_with_range(range.first, range.second));
      ARROW_ASSIGN_OR_RAISE(auto rbs, rbreader->ToRecordBatches());
      // append rbs to rbs_in_file
      std::move(rbs.begin(), rbs.end(), std::back_inserter(rbs_in_file));
    }

    // generate chunk_rb_map
    size_t rbs_idx = 0;
    size_t rbs_offset = 0;
    for (size_t i = 0; i < chunk_idxs.size(); ++i) {
      const auto& chunk_info = chunk_infos_[chunk_idxs[i]];
      if (UNLIKELY(((rbs_in_file[rbs_idx]->num_rows() - rbs_offset) < chunk_info.number_of_rows))) {
        return arrow::Status::Invalid("Invalid slice of record batchs: ", std::to_string(chunk_info.number_of_rows),
                                      " out of " + std::to_string(rbs_in_file[rbs_idx]->num_rows() - rbs_offset),
                                      ", [chunk info=", chunk_info.ToString(), "]");
      }

      auto rb = rbs_in_file[rbs_idx]->Slice(rbs_offset, chunk_info.number_of_rows);
      chunk_rb_map[chunk_idxs[i]] = rb;
      rbs_offset += chunk_info.number_of_rows;

      assert(rbs_offset <= rbs_in_file[rbs_idx]->num_rows());
      if (rbs_offset == rbs_in_file[rbs_idx]->num_rows()) {
        rbs_idx++;
        rbs_offset = 0;
      }
    }
  }

  // 3. generate result
  std::vector<std::shared_ptr<arrow::RecordBatch>> result;
  for (const auto& chunk_idx : chunk_indices) {
    assert(chunk_rb_map.find(chunk_idx) != chunk_rb_map.end());
    result.emplace_back(chunk_rb_map[chunk_idx]);
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> ColumnGroupReaderImpl::take(
    const std::vector<int64_t>& row_indices) {
  assert(!format_readers_.empty());
  return arrow::Status::NotImplemented("take is not implemented");
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_size(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].avg_memory_size;
}

arrow::Result<uint64_t> ColumnGroupReaderImpl::get_chunk_rows(int64_t chunk_index) {
  assert(!format_readers_.empty());
  if (UNLIKELY(chunk_index < 0 || chunk_index >= chunk_infos_.size())) {
    return arrow::Status::Invalid("Chunk index out of range: " + std::to_string(chunk_index) + " out of " +
                                  std::to_string(chunk_infos_.size()));
  }
  return chunk_infos_[chunk_index].number_of_rows;
}

}  // namespace milvus_storage::api
