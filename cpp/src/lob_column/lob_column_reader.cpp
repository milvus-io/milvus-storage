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
  public:
  LobColumnReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const LobColumnConfig& config)
      : fs_(std::move(fs)), config_(config), closed_(false) {}

  ~LobColumnReaderImpl() override {
    if (!closed_) {
      // best effort close, ignore errors in destructor
      (void)Close();
    }
  }

  arrow::Result<std::vector<uint8_t>> ReadData(const uint8_t* encoded_ref, size_t ref_size) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

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
    ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(file_id_str));

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
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

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
      ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(file_id_str));

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

  arrow::Result<std::shared_ptr<arrow::BinaryArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

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
      ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(file_id_str));

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
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(file_id_str));

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
    if (closed_) {
      return arrow::Status::OK();
    }

    reader_cache_.clear();
    closed_ = true;
    return arrow::Status::OK();
  }

  bool IsClosed() const override { return closed_; }

  void ClearCache() override { reader_cache_.clear(); }

  private:
  std::shared_ptr<arrow::DataType> ArrowType() const {
    return config_.data_type == LobDataType::kText ? arrow::utf8() : arrow::binary();
  }

  std::string FieldName() const { return config_.data_type == LobDataType::kText ? "text_data" : "binary_data"; }

  // get or open a vortex reader for the given file_id_str (UUID string)
  arrow::Result<std::shared_ptr<vortex::VortexFormatReader>> GetOrOpenReader(const std::string& file_id_str) {
    // check cache
    auto it = reader_cache_.find(file_id_str);
    if (it != reader_cache_.end()) {
      return it->second;
    }

    // build file path using the function from lob_reference.h
    auto file_path = BuildLOBFilePath(config_.lob_base_path, file_id_str);

    auto field_name = FieldName();
    auto schema = arrow::schema({
        arrow::field(field_name, ArrowType(), false),
    });

    // create and open reader
    auto reader = std::make_shared<vortex::VortexFormatReader>(fs_, schema, file_path, config_.properties,
                                                               std::vector<std::string>{field_name});

    ARROW_RETURN_NOT_OK(reader->open());

    // cache the reader
    reader_cache_[file_id_str] = reader;
    return reader;
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  LobColumnConfig config_;
  bool closed_;

  // cache of open vortex readers, keyed by file_id
  std::unordered_map<std::string, std::shared_ptr<vortex::VortexFormatReader>> reader_cache_;
};

// factory function to create LobColumnReader
arrow::Result<std::unique_ptr<LobColumnReader>> CreateLobColumnReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                      const LobColumnConfig& config) {
  return std::make_unique<LobColumnReaderImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::lob_column
