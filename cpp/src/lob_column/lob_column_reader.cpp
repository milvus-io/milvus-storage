// Copyright 2024 Zilliz
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

#include "milvus-storage/lob_column/lob_column_reader.h"

#include <arrow/array/builder_binary.h>
#include <arrow/table.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <folly/futures/Future.h>
#include <folly/Synchronized.h>
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/arrow_util.h"
#include <unordered_map>

#include "milvus-storage/format/vortex/vortex_format_reader.h"

namespace milvus_storage::lob_column {

namespace {

// zero-copy view into an Arrow buffer
struct DataView {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// extract raw bytes from a BinaryArray chunk at the given index
std::vector<uint8_t> GetBinaryValue(const std::shared_ptr<arrow::BinaryArray>& array, int64_t index) {
  int32_t length;
  const uint8_t* value = array->GetValue(index, &length);
  return std::vector<uint8_t>(value, value + length);
}

}  // namespace

// implementation of LobColumnReader using Vortex format
class LobColumnReaderImpl : public LobColumnReader {
  using BatchResult = arrow::Result<std::vector<std::string>>;
  using TableResult = arrow::Result<std::shared_ptr<arrow::Table>>;

  using FileGroups = std::map<std::string, std::vector<std::pair<size_t, int32_t>>>;
  using ReaderCache =
      folly::Synchronized<std::unordered_map<std::string, std::shared_ptr<vortex::VortexFormatReader>>, std::mutex>;

  public:
  LobColumnReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const LobColumnConfig& config)
      : fs_(std::move(fs)), config_(config), reader_cache_(std::make_shared<ReaderCache>()) {}

  ~LobColumnReaderImpl() override = default;

  arrow::Result<std::vector<uint8_t>> ReadData(const uint8_t* encoded_ref, size_t ref_size) override {
    ARROW_ASSIGN_OR_RAISE(auto cache, GetReadCache());

    if (ref_size == 0 || encoded_ref == nullptr) {
      return arrow::Status::Invalid("invalid encoded reference");
    }

    // check if inline or LOB reference
    if (IsInlineData(encoded_ref)) {
      const uint8_t* payload;
      size_t payload_size;
      DecodeInlineData(encoded_ref, ref_size, &payload, &payload_size);
      if (payload == nullptr) {
        return std::vector<uint8_t>{};
      }
      return std::vector<uint8_t>(payload, payload + payload_size);
    }

    if (ref_size != LOB_REFERENCE_SIZE) {
      return arrow::Status::Invalid("invalid LOB reference size: ", ref_size, ", expected: ", LOB_REFERENCE_SIZE);
    }

    auto [file_id_str, row_offset] = DecodeLOBReference(encoded_ref);

    // get or open the vortex reader for this file
    ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(fs_, config_, cache, file_id_str));

    // read using take API
    std::vector<int64_t> indices = {static_cast<int64_t>(row_offset)};
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices));

    if (table->num_rows() == 0) {
      return arrow::Status::IndexError("row offset out of range: ", row_offset);
    }

    auto lob_column = table->column(0);
    if (lob_column->num_chunks() == 0 || lob_column->chunk(0)->length() == 0) {
      return arrow::Status::IndexError("no data at row offset: ", row_offset);
    }

    auto binary_array = std::static_pointer_cast<arrow::BinaryArray>(lob_column->chunk(0));
    return GetBinaryValue(binary_array, 0);
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> ReadBatchData(const std::vector<EncodedRef>& encoded_refs) override {
    ARROW_ASSIGN_OR_RAISE(auto cache, GetReadCache());

    std::vector<std::vector<uint8_t>> results(encoded_refs.size());

    // group LOB references by file_id for efficient batch reading
    // key: file_id (as string), value: vector of (original_index, row_offset)
    std::map<std::string, std::vector<std::pair<size_t, int32_t>>> file_groups;

    for (size_t i = 0; i < encoded_refs.size(); i++) {
      const auto& ref = encoded_refs[i];

      if (ref.data == nullptr || ref.size == 0) {
        results[i] = {};
        continue;
      }

      if (IsInlineData(ref.data)) {
        const uint8_t* payload;
        size_t payload_size;
        DecodeInlineData(ref.data, ref.size, &payload, &payload_size);
        if (payload != nullptr) {
          results[i] = std::vector<uint8_t>(payload, payload + payload_size);
        }
      } else {
        if (ref.size != LOB_REFERENCE_SIZE) {
          return arrow::Status::Invalid("invalid LOB reference size at index ", i);
        }

        auto [file_id_str, row_offset] = DecodeLOBReference(ref.data);
        file_groups[file_id_str].emplace_back(i, row_offset);
      }
    }

    // process each file group
    for (auto& [file_id_str, idx_offsets] : file_groups) {
      ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(fs_, config_, cache, file_id_str));

      // collect row indices and sort for efficient access
      std::vector<std::pair<int32_t, size_t>> sorted_offsets;  // (row_offset, original_group_index)
      sorted_offsets.reserve(idx_offsets.size());
      for (size_t j = 0; j < idx_offsets.size(); j++) {
        sorted_offsets.emplace_back(idx_offsets[j].second, j);
      }
      std::sort(sorted_offsets.begin(), sorted_offsets.end());

      // build row indices array for take
      std::vector<int64_t> row_indices;
      row_indices.reserve(sorted_offsets.size());
      for (const auto& [offset, _] : sorted_offsets) {
        row_indices.push_back(static_cast<int64_t>(offset));
      }

      // batch read using take API
      ARROW_ASSIGN_OR_RAISE(auto table, reader->take(row_indices));

      // extract data and map back to original order
      auto lob_column = table->column(0);
      size_t data_idx = 0;

      for (int chunk_idx = 0; chunk_idx < lob_column->num_chunks(); chunk_idx++) {
        auto binary_array = std::static_pointer_cast<arrow::BinaryArray>(lob_column->chunk(chunk_idx));
        for (int64_t k = 0; k < binary_array->length(); k++) {
          if (data_idx < sorted_offsets.size()) {
            size_t group_idx = sorted_offsets[data_idx].second;
            size_t original_idx = idx_offsets[group_idx].first;
            results[original_idx] = GetBinaryValue(binary_array, k);
            data_idx++;
          }
        }
      }
    }

    return results;
  }

  folly::SemiFuture<BatchResult> ReadBatchAsync(const std::vector<EncodedRef>& encoded_refs) override {
    FOLLY_ARROW_ASSIGN_OR_RAISE(auto cache, GetReadCache());
    std::vector<std::string> results(encoded_refs.size());
    FileGroups files;
    // Copy inline bytes and decode file references before returning. No caller
    // buffers are retained by the deferred work.
    for (size_t i = 0; i < encoded_refs.size(); ++i) {
      const auto& ref = encoded_refs[i];
      if (!ref.data || ref.size == 0)
        continue;
      if (IsInlineData(ref.data)) {
        const uint8_t* payload;
        size_t size;
        DecodeInlineData(ref.data, ref.size, &payload, &size);
        if (payload)
          results[i].assign(reinterpret_cast<const char*>(payload), size);
      } else {
        if (ref.size != LOB_REFERENCE_SIZE) {
          return folly::makeSemiFuture(BatchResult(arrow::Status::Invalid("invalid LOB reference size at index ", i)));
        }
        auto ref_value = DecodeLOBReference(ref.data);
        files[ref_value.file_id].emplace_back(i, ref_value.row_offset);
      }
    }
    // The caller selects the executor with via(). Even cold opens run there,
    // not on the thread constructing this future or a storage-owned C++ pool.
    return folly::makeSemiFuture().deferValue([fs = fs_, config = config_, cache = std::move(cache),
                                               files = std::move(files),
                                               results = std::move(results)](folly::Unit) mutable {
      return ReadBatchFilesAsync(fs, config, cache, std::move(files), std::move(results));
    });
  }

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) override {
    ARROW_ASSIGN_OR_RAISE(auto cache, GetReadCache());

    const int64_t n = encoded_refs->length();

    std::vector<DataView> views(n);
    // vortex tables must stay alive until the builder copies their data
    std::vector<std::shared_ptr<arrow::Table>> keep_alive;

    // group LOB references by file_id for batch reading
    std::map<std::string, std::vector<std::pair<int64_t, int32_t>>> file_groups;

    size_t total_bytes = 0;
    for (int64_t i = 0; i < n; i++) {
      if (encoded_refs->IsNull(i)) {
        continue;
      }

      int32_t length;
      const uint8_t* raw = encoded_refs->GetValue(i, &length);

      if (IsInlineData(raw)) {
        const uint8_t* payload;
        size_t payload_size;
        DecodeInlineData(raw, static_cast<size_t>(length), &payload, &payload_size);
        views[i] = {payload, payload_size};
        total_bytes += payload_size;
      } else {
        if (static_cast<size_t>(length) != LOB_REFERENCE_SIZE) {
          return arrow::Status::Invalid("invalid LOB reference size at index ", i);
        }
        auto ref = DecodeLOBReference(raw);
        file_groups[ref.file_id].emplace_back(i, ref.row_offset);
      }
    }

    // batch read each file's LOB entries via vortex take()
    for (auto& [file_id_str, idx_offsets] : file_groups) {
      ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(fs_, config_, cache, file_id_str));

      // sort by row_offset for sequential I/O
      std::vector<std::pair<int32_t, size_t>> sorted;
      sorted.reserve(idx_offsets.size());
      for (size_t j = 0; j < idx_offsets.size(); j++) {
        sorted.emplace_back(idx_offsets[j].second, j);
      }
      std::sort(sorted.begin(), sorted.end());

      std::vector<int64_t> row_indices;
      row_indices.reserve(sorted.size());
      for (const auto& [offset, _] : sorted) {
        row_indices.push_back(static_cast<int64_t>(offset));
      }

      ARROW_ASSIGN_OR_RAISE(auto table, reader->take(row_indices));
      keep_alive.push_back(table);

      // populate views with zero-copy pointers into the table's chunks
      auto lob_col = table->column(0);
      size_t data_idx = 0;
      for (int c = 0; c < lob_col->num_chunks(); c++) {
        auto chunk = std::static_pointer_cast<arrow::BinaryArray>(lob_col->chunk(c));
        for (int64_t k = 0; k < chunk->length() && data_idx < sorted.size(); k++, data_idx++) {
          size_t group_idx = sorted[data_idx].second;
          int64_t original_idx = idx_offsets[group_idx].first;

          int32_t val_len;
          const uint8_t* val_data = chunk->GetValue(k, &val_len);
          views[original_idx] = {val_data, static_cast<size_t>(val_len)};
          total_bytes += val_len;
        }
      }
    }

    // single copy: views → BinaryBuilder
    arrow::BinaryBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(n));
    ARROW_RETURN_NOT_OK(builder.ReserveData(static_cast<int64_t>(total_bytes)));

    for (int64_t i = 0; i < n; i++) {
      if (encoded_refs->IsNull(i)) {
        ARROW_RETURN_NOT_OK(builder.AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(builder.Append(views[i].data, views[i].size));
      }
    }

    std::shared_ptr<arrow::BinaryArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    return result;
  }

  arrow::Result<std::vector<std::vector<uint8_t>>> TakeData(const std::string& file_id_str,
                                                            const std::vector<int32_t>& row_offsets) override {
    ARROW_ASSIGN_OR_RAISE(auto cache, GetReadCache());

    ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(fs_, config_, cache, file_id_str));

    // convert to int64_t for take API
    std::vector<int64_t> indices;
    indices.reserve(row_offsets.size());
    for (auto offset : row_offsets) {
      indices.push_back(static_cast<int64_t>(offset));
    }

    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices));

    std::vector<std::vector<uint8_t>> results;
    results.reserve(row_offsets.size());

    auto lob_column = table->column(0);
    for (int chunk_idx = 0; chunk_idx < lob_column->num_chunks(); chunk_idx++) {
      auto binary_array = std::static_pointer_cast<arrow::BinaryArray>(lob_column->chunk(chunk_idx));
      for (int64_t k = 0; k < binary_array->length(); k++) {
        results.push_back(GetBinaryValue(binary_array, k));
      }
    }

    return results;
  }

  arrow::Status Close() override {
    std::shared_ptr<ReaderCache> retired;
    {
      std::lock_guard lock(state_mutex_);
      retired = std::move(reader_cache_);
    }
    // Reclaim resources outside the lock. Requests retain their own references.
    return arrow::Status::OK();
  }

  bool IsClosed() const override {
    std::lock_guard lock(state_mutex_);
    return !reader_cache_;
  }

  void ClearCache() override {
    std::shared_ptr<ReaderCache> retired;
    {
      std::lock_guard lock(state_mutex_);
      if (!reader_cache_)
        return;
      auto fresh = std::make_shared<ReaderCache>();
      retired = std::exchange(reader_cache_, std::move(fresh));
    }
  }

  private:
  static folly::SemiFuture<BatchResult> ReadBatchFilesAsync(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                            const LobColumnConfig& config,
                                                            const std::shared_ptr<ReaderCache>& cache,
                                                            FileGroups files,
                                                            std::vector<std::string> results) {
    if (files.empty()) {
      return folly::makeSemiFuture(BatchResult(std::move(results)));
    }
    struct FileTask {
      std::shared_ptr<vortex::VortexFormatReader> reader;
      std::vector<int64_t> rows;
      std::vector<std::vector<size_t>> positions;
    };
    std::vector<FileTask> tasks;
    tasks.reserve(files.size());
    // Open all requested files before any scans are submitted, reusing the same
    // cache as synchronous reads. A failed open cannot abandon a running scan.
    for (auto& [file_id, offsets] : files) {
      FOLLY_ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(fs, config, cache, file_id));
      std::sort(offsets.begin(), offsets.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
      FileTask task{std::move(reader), {}, {}};
      for (const auto& [position, offset] : offsets) {
        if (task.rows.empty() || task.rows.back() != offset) {
          task.rows.push_back(offset);
          task.positions.emplace_back();
        }
        task.positions.back().push_back(position);
      }
      tasks.push_back(std::move(task));
    }

    std::vector<folly::SemiFuture<TableResult>> futures;
    futures.reserve(tasks.size());
    for (auto& task : tasks) {
      // Defer submission itself: exceptions become failed futures and all
      // siblings still participate in collectAll. Retain each file reader
      // through completion of its native scan.
      futures.push_back(
          folly::makeSemiFuture().deferValue([reader = task.reader, rows = std::move(task.rows)](folly::Unit) {
            return reader->take_async(rows).deferValue([reader](TableResult result) {
              (void)reader;  // Lifetime-only capture.
              return result;
            });
          }));
    }
    return folly::collectAll(std::move(futures))
        .deferValue([results = std::move(results), tasks = std::move(tasks)](auto&& completed) mutable -> BatchResult {
          // collectAll completes every scan before propagating any Arrow error
          // or backend exception. Result assembly runs on the caller executor.
          for (size_t i = 0; i < tasks.size(); ++i) {
            ARROW_ASSIGN_OR_RAISE(auto table, std::move(completed[i]).value());
            const auto& positions = tasks[i].positions;
            if (!table || table->num_columns() != 1 || table->num_rows() != static_cast<int64_t>(positions.size())) {
              return MakeExtendError(ExtendStatusCode::PackedUnexpected, "LOB take returned an unexpected shape");
            }
            size_t row = 0;
            for (const auto& chunk : table->column(0)->chunks()) {
              if (chunk->type_id() != arrow::Type::STRING && chunk->type_id() != arrow::Type::BINARY) {
                return MakeExtendError(ExtendStatusCode::PackedUnexpected, "LOB take returned an unexpected type");
              }
              auto binary = std::static_pointer_cast<arrow::BinaryArray>(chunk);
              for (int64_t k = 0; k < binary->length(); ++k) {
                if (binary->IsNull(k)) {
                  return MakeExtendError(ExtendStatusCode::PackedUnexpected, "LOB take returned an unexpected null");
                }
                auto value = binary->GetString(k);
                const auto& destinations = positions[row++];
                for (size_t j = 0; j + 1 < destinations.size(); ++j) results[destinations[j]] = value;
                results[destinations.back()] = std::move(value);
              }
            }
          }
          return std::move(results);
        });
  }

  static arrow::Result<std::shared_ptr<vortex::VortexFormatReader>> GetOrOpenReader(
      const std::shared_ptr<arrow::fs::FileSystem>& fs,
      const LobColumnConfig& config,
      const std::shared_ptr<ReaderCache>& cache,
      const std::string& file_id) {
    {
      auto locked = cache->lock();
      auto it = locked->find(file_id);
      if (it != locked->end())
        return it->second;
    }
    // Only cache access is locked. Concurrent misses may open duplicate files;
    // publication below reuses the winner, without holding a lock across I/O.
    const bool text = config.data_type == LobDataType::kText;
    const std::string field_name = text ? "text_data" : "binary_data";
    auto schema = arrow::schema({arrow::field(field_name, text ? arrow::utf8() : arrow::binary(), false)});
    auto reader =
        std::make_shared<vortex::VortexFormatReader>(fs, schema, BuildLOBFilePath(config.lob_base_path, file_id),
                                                     config.properties, std::vector<std::string>{field_name});
    ARROW_RETURN_NOT_OK(reader->open());
    {
      auto locked = cache->lock();
      return locked->emplace(file_id, reader).first->second;
    }
  }

  arrow::Result<std::shared_ptr<ReaderCache>> GetReadCache() const {
    std::lock_guard lock(state_mutex_);
    if (!reader_cache_)
      return arrow::Status::Invalid("reader is closed");
    return reader_cache_;
  }

  const std::shared_ptr<arrow::fs::FileSystem> fs_;
  const LobColumnConfig config_;
  mutable std::mutex state_mutex_;
  // Each request retains this cache snapshot. ClearCache swaps in an empty map;
  // old in-flight opens can only populate the retired map. nullptr means closed.
  std::shared_ptr<ReaderCache> reader_cache_;
};

// factory function to create LobColumnReader
arrow::Result<std::unique_ptr<LobColumnReader>> CreateLobColumnReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                      const LobColumnConfig& config) {
  return std::make_unique<LobColumnReaderImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::lob_column
