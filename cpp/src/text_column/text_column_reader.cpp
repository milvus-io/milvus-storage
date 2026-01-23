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

#include "milvus-storage/text_column/text_column_reader.h"

#include <arrow/array/builder_binary.h>
#include <arrow/table.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>

#include "milvus-storage/common/log.h"

#ifdef BUILD_VORTEX_BRIDGE
#include "milvus-storage/format/vortex/vortex_format_reader.h"
#endif

namespace milvus_storage::text_column {

#ifdef BUILD_VORTEX_BRIDGE

// implementation of TextColumnReader using Vortex format
class TextColumnReaderImpl : public TextColumnReader {
  public:
  TextColumnReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const TextColumnConfig& config)
      : fs_(std::move(fs)), config_(config), closed_(false) {}

  ~TextColumnReaderImpl() override {
    if (!closed_) {
      // best effort close, ignore errors in destructor
      (void)Close();
    }
  }

  arrow::Result<std::string> ReadText(const uint8_t* encoded_ref, size_t ref_size) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    if (ref_size == 0 || encoded_ref == nullptr) {
      return arrow::Status::Invalid("invalid encoded reference");
    }

    // check if inline or LOB reference
    if (IsInlineData(encoded_ref)) {
      return DecodeInlineText(encoded_ref, ref_size);
    }

    // LOB reference - must be exactly 44 bytes
    if (ref_size != LOB_REFERENCE_SIZE) {
      return arrow::Status::Invalid("invalid LOB reference size: ", ref_size, ", expected: ", LOB_REFERENCE_SIZE);
    }

    // decode and read from file
    std::string file_id_str;
    int32_t row_offset;
    DecodeLOBReference(encoded_ref, &file_id_str, &row_offset);

    // get or open the vortex reader for this file
    ARROW_ASSIGN_OR_RAISE(auto reader, GetOrOpenReader(file_id_str));

    // read the text using take API
    std::vector<int64_t> indices = {static_cast<int64_t>(row_offset)};
    ARROW_ASSIGN_OR_RAISE(auto table, reader->take(indices));

    // extract text from result
    if (table->num_rows() == 0) {
      return arrow::Status::IndexError("row offset out of range: ", row_offset);
    }

    auto text_column = table->column(0);
    if (text_column->num_chunks() == 0 || text_column->chunk(0)->length() == 0) {
      return arrow::Status::IndexError("no data at row offset: ", row_offset);
    }

    auto string_array = std::static_pointer_cast<arrow::StringArray>(text_column->chunk(0));
    return string_array->GetString(0);
  }

  arrow::Result<std::vector<std::string>> ReadBatch(const std::vector<EncodedRef>& encoded_refs) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    std::vector<std::string> results(encoded_refs.size());

    // group LOB references by file_id for efficient batch reading
    // key: file_id (as string), value: vector of (original_index, row_offset)
    std::map<std::string, std::vector<std::pair<size_t, int32_t>>> file_groups;

    for (size_t i = 0; i < encoded_refs.size(); i++) {
      const auto& ref = encoded_refs[i];

      if (ref.data == nullptr || ref.size == 0) {
        results[i] = "";
        continue;
      }

      if (IsInlineData(ref.data)) {
        // inline text - decode directly
        results[i] = DecodeInlineText(ref.data, ref.size);
      } else {
        // LOB reference - group by file_id_str
        if (ref.size != LOB_REFERENCE_SIZE) {
          return arrow::Status::Invalid("invalid LOB reference size at index ", i);
        }

        std::string file_id_str;
        int32_t row_offset;
        DecodeLOBReference(ref.data, &file_id_str, &row_offset);

        file_groups[file_id_str].emplace_back(i, row_offset);
      }
    }

    // process each file group
    for (auto& [file_id_str, idx_offsets] : file_groups) {
      // get or open the reader
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

      // extract texts and map back to original order
      auto text_column = table->column(0);
      size_t text_idx = 0;

      for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
        auto string_array = std::static_pointer_cast<arrow::StringArray>(text_column->chunk(chunk_idx));
        for (int64_t k = 0; k < string_array->length(); k++) {
          if (text_idx < sorted_offsets.size()) {
            size_t group_idx = sorted_offsets[text_idx].second;
            size_t original_idx = idx_offsets[group_idx].first;
            results[original_idx] = string_array->GetString(k);
            text_idx++;
          }
        }
      }
    }

    return results;
  }

  arrow::Result<std::shared_ptr<arrow::StringArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) override {
    if (closed_) {
      return arrow::Status::Invalid("reader is closed");
    }

    // collect references with size info
    std::vector<EncodedRef> refs;
    refs.reserve(encoded_refs->length());

    for (int64_t i = 0; i < encoded_refs->length(); i++) {
      if (encoded_refs->IsNull(i)) {
        refs.push_back({nullptr, 0});
      } else {
        int32_t length;
        const uint8_t* data = encoded_refs->GetValue(i, &length);
        refs.push_back({data, static_cast<size_t>(length)});
      }
    }

    // read all texts
    ARROW_ASSIGN_OR_RAISE(auto texts, ReadBatch(refs));

    // build string array
    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(texts.size()));

    for (size_t i = 0; i < texts.size(); i++) {
      if (encoded_refs->IsNull(i)) {
        ARROW_RETURN_NOT_OK(builder.AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(builder.Append(texts[i]));
      }
    }

    std::shared_ptr<arrow::StringArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    return result;
  }

  arrow::Result<std::vector<std::string>> Take(const std::string& file_id_str,
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

    // extract texts
    std::vector<std::string> results;
    results.reserve(row_offsets.size());

    auto text_column = table->column(0);
    for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
      auto string_array = std::static_pointer_cast<arrow::StringArray>(text_column->chunk(chunk_idx));
      for (int64_t k = 0; k < string_array->length(); k++) {
        results.push_back(string_array->GetString(k));
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
  // get or open a vortex reader for the given file_id_str (UUID string)
  arrow::Result<std::shared_ptr<vortex::VortexFormatReader>> GetOrOpenReader(const std::string& file_id_str) {
    // check cache
    auto it = reader_cache_.find(file_id_str);
    if (it != reader_cache_.end()) {
      return it->second;
    }

    // build file path using the function from lob_reference.h
    auto file_path = BuildLOBFilePath(config_.lob_base_path, file_id_str);

    // create schema (matches writer schema)
    auto schema = arrow::schema({
        arrow::field("text_data", arrow::utf8(), false),
    });

    // create and open reader
    auto reader = std::make_shared<vortex::VortexFormatReader>(fs_, schema, file_path, config_.properties,
                                                               std::vector<std::string>{"text_data"});

    ARROW_RETURN_NOT_OK(reader->open());

    // cache the reader
    reader_cache_[file_id_str] = reader;
    return reader;
  }

  private:
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  TextColumnConfig config_;
  bool closed_;

  // cache of open vortex readers, keyed by file_id
  std::unordered_map<std::string, std::shared_ptr<vortex::VortexFormatReader>> reader_cache_;
};

#else  // BUILD_VORTEX_BRIDGE

// stub implementation when Vortex is not available
class TextColumnReaderImpl : public TextColumnReader {
  public:
  TextColumnReaderImpl(std::shared_ptr<arrow::fs::FileSystem> fs, const TextColumnConfig& config) {}

  arrow::Result<std::string> ReadText(const uint8_t* encoded_ref, size_t ref_size) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::vector<std::string>> ReadBatch(const std::vector<EncodedRef>& encoded_refs) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::shared_ptr<arrow::StringArray>> ReadArrowArray(
      const std::shared_ptr<arrow::BinaryArray>& encoded_refs) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Result<std::vector<std::string>> Take(const std::string& file_id_str,
                                               const std::vector<int32_t>& row_offsets) override {
    return arrow::Status::NotImplemented("Vortex support is not enabled");
  }

  arrow::Status Close() override { return arrow::Status::NotImplemented("Vortex support is not enabled"); }

  bool IsClosed() const override { return true; }

  void ClearCache() override {}
};

#endif  // BUILD_VORTEX_BRIDGE

// factory function to create TextColumnReader
arrow::Result<std::unique_ptr<TextColumnReader>> CreateTextColumnReader(std::shared_ptr<arrow::fs::FileSystem> fs,
                                                                        const TextColumnConfig& config) {
  return std::make_unique<TextColumnReaderImpl>(std::move(fs), config);
}

}  // namespace milvus_storage::text_column
