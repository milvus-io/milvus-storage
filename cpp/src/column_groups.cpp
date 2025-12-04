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

#include "milvus-storage/column_groups.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <set>

#include <arrow/status.h>
#include <arrow/result.h>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Specific.hh>
#include <avro/Stream.hh>

#include "milvus-storage/common/macro.h"

namespace milvus_storage::api {

ColumnGroups::ColumnGroups(std::vector<std::shared_ptr<ColumnGroup>> column_groups)
    : column_groups_(std::move(column_groups)) {
  rebuild_column_mapping();
}

arrow::Result<std::string> ColumnGroups::serialize() const {
  try {
    std::ostringstream oss;
    std::unique_ptr<avro::OutputStream> out = avro::ostreamOutputStream(oss);
    avro::EncoderPtr encoder = avro::binaryEncoder();
    encoder->init(*out);

    // Encode column_groups
    encoder->arrayStart();
    if (!column_groups_.empty()) {
      encoder->setItemCount(column_groups_.size());
      for (const auto& group : column_groups_) {
        encoder->startItem();
        if (group) {
          avro::encode(*encoder, group->columns);

          // Encode files
          encoder->arrayStart();
          encoder->setItemCount(group->files.size());
          for (const auto& file : group->files) {
            encoder->startItem();
            avro::encode(*encoder, file.path);
            avro::encode(*encoder, file.start_index);
            avro::encode(*encoder, file.end_index);

            bool has_private_data = file.private_data.has_value();
            avro::encode(*encoder, has_private_data);
            if (has_private_data) {
              avro::encode(*encoder, file.private_data.value());
            }
          }
          encoder->arrayEnd();

          avro::encode(*encoder, group->format);
        } else {
          // Encode empty fields for null group
          avro::encode(*encoder, std::vector<std::string>());

          // Empty files array
          encoder->arrayStart();
          encoder->setItemCount(0);
          encoder->arrayEnd();

          avro::encode(*encoder, std::string());
        }
      }
    }
    encoder->arrayEnd();

    // Encode metadata map
    encoder->mapStart();
    if (!metadata_.empty()) {
      encoder->setItemCount(metadata_.size());
      for (const auto& [key, val] : metadata_) {
        encoder->startItem();
        avro::encode(*encoder, key);
        avro::encode(*encoder, val);
      }
    }
    encoder->mapEnd();

    encoder->flush();
    return oss.str();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid("Failed to serialize ColumnGroups: " + std::string(e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Failed to serialize ColumnGroups (std::exception): " + std::string(e.what()));
  }
}

arrow::Status ColumnGroups::deserialize(const std::string_view& data) {
  if (data.empty()) {
    return arrow::Status::Invalid("Empty data for deserialization");
  }
  try {
    std::unique_ptr<avro::InputStream> in =
        avro::memoryInputStream(reinterpret_cast<const uint8_t*>(data.data()), data.size());
    avro::DecoderPtr decoder = avro::binaryDecoder();
    decoder->init(*in);

    // Decode column_groups
    column_groups_.clear();
    size_t n = decoder->arrayStart();

    // Sanity check
    if (n > data.size()) {
      return arrow::Status::Invalid("Too many column groups in manifest: " + std::to_string(n));
    }
    column_groups_.reserve(n);

    while (n != 0) {
      for (size_t i = 0; i < n; ++i) {
        auto group = std::make_shared<ColumnGroup>();

        avro::decode(*decoder, group->columns);

        // Decode files
        size_t file_count = decoder->arrayStart();
        if (file_count > data.size()) {
          return arrow::Status::Invalid("Too many files in column group");
        }

        while (file_count != 0) {
          for (size_t k = 0; k < file_count; ++k) {
            ColumnGroupFile file;
            avro::decode(*decoder, file.path);
            avro::decode(*decoder, file.start_index);
            avro::decode(*decoder, file.end_index);

            bool has_private_data;
            avro::decode(*decoder, has_private_data);
            if (has_private_data) {
              std::vector<uint8_t> val;
              avro::decode(*decoder, val);
              file.private_data = std::move(val);
            }
            group->files.emplace_back(std::move(file));
          }
          file_count = decoder->arrayNext();
        }

        avro::decode(*decoder, group->format);
        column_groups_.push_back(group);
      }
      n = decoder->arrayNext();
    }

    // Extract metadata
    metadata_.clear();
    n = decoder->mapStart();
    if (n > data.size()) {
      return arrow::Status::Invalid("Too many metadata entries in manifest: " + std::to_string(n));
    }
    metadata_.reserve(n);
    while (n != 0) {
      for (size_t i = 0; i < n; ++i) {
        std::string key;
        avro::decode(*decoder, key);
        std::string val;
        avro::decode(*decoder, val);
        metadata_.emplace_back(key, val);
      }
      n = decoder->mapNext();
    }

    rebuild_column_mapping();
    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    column_groups_.clear();
    metadata_.clear();
    return arrow::Status::Invalid("Failed to deserialize ColumnGroups: " + std::string(e.what()));
  } catch (const std::exception& e) {
    column_groups_.clear();
    metadata_.clear();
    return arrow::Status::Invalid("Failed to deserialize ColumnGroups (std::exception): " + std::string(e.what()));
  }
}

std::string ColumnGroups::to_string() const {
  std::string result;
  auto serialize_result = serialize();
  if (serialize_result.ok()) {
    result = serialize_result.ValueOrDie();
  } else {
    result = "Failed to serialize ColumnGroups: " + serialize_result.status().ToString();
  }
  return result;
}

void ColumnGroups::rebuild_column_mapping() {
  column_to_group_map_.clear();

  for (size_t i = 0; i < column_groups_.size(); i++) {
    auto& cg = column_groups_[i];
    for (const auto& column_name : cg->columns) {
      column_to_group_map_[column_name] = i;
    }
  }
}

std::vector<std::shared_ptr<ColumnGroup>> ColumnGroups::get_all() const { return column_groups_; }

arrow::Result<std::pair<std::string_view, std::string_view>> ColumnGroups::get_metadata(size_t idx) const {
  if (idx >= metadata_.size()) {
    return arrow::Status::Invalid("Metadata index out of range: " + std::to_string(idx));
  }

  return std::make_pair(std::string_view(metadata_[idx].first), std::string_view(metadata_[idx].second));
}

std::shared_ptr<ColumnGroup> ColumnGroups::get_column_group(const std::string& column_name) const {
  auto it = column_to_group_map_.find(column_name);
  if (it != column_to_group_map_.end()) {
    return column_groups_[it->second];
  }

  return nullptr;
}

std::shared_ptr<ColumnGroup> ColumnGroups::get_column_group(size_t column_group_index) const {
  if (column_group_index < column_groups_.size()) {
    return column_groups_[column_group_index];
  }

  return nullptr;
}

arrow::Status ColumnGroups::append_files(const std::shared_ptr<ColumnGroups>& new_cg) {
  auto& column_groups1 = column_groups_;
  auto& column_groups2 = new_cg->column_groups_;

  // if no existing column groups, directly assign
  // this situation happens when no data has been written yet
  if (column_groups1.empty()) {
    column_groups_ = column_groups2;
    rebuild_column_mapping();
    return arrow::Status::OK();
  }

  if (column_groups1.size() != column_groups2.size()) {
    return arrow::Status::Invalid("Column group size mismatch");
  }

  for (size_t i = 0; i < column_groups1.size(); i++) {
    auto& cg1 = column_groups1[i];
    auto& cg2 = column_groups2[i];

    // only compare columns and format, no need match the paths
    if (cg1->columns.size() != cg2->columns.size()) {
      return arrow::Status::Invalid("Column groups size mismatch at index ", std::to_string(i));
    }

    // compare format
    if (cg1->format != cg2->format) {
      return arrow::Status::Invalid("Column groups format mismatch at index ", std::to_string(i));
    }

    // check columns
    std::set<std::string_view> col_set(cg1->columns.begin(), cg1->columns.end());
    for (const auto& col : cg2->columns) {
      if (col_set.find(col) == col_set.end()) {
        return arrow::Status::Invalid("Column group columns mismatch at index ", std::to_string(i));
      }
    }

    // no need check field `files`, we allow overlap paths in this field
    // merge field `files`
    for (const auto& file : cg2->files) {
      cg1->files.emplace_back(file);
    }
  }

  return arrow::Status::OK();
}

arrow::Status ColumnGroups::add_column_group(std::shared_ptr<ColumnGroup> column_group) {
  if (!column_group) {
    return arrow::Status::Invalid("column group is empty");
  }

  // Check for column conflicts with existing column groups
  for (const auto& column_name : column_group->columns) {
    auto existing_cg = get_column_group(column_name);
    if (existing_cg != nullptr) {
      return arrow::Status::Invalid("Column '" + column_name + "' already exists in another column group");
    }
  }

  // Add the column group
  column_groups_.emplace_back(std::move(column_group));

  // Update column mapping
  for (const auto& column_name : column_groups_.back()->columns) {
    column_to_group_map_[column_name] = column_groups_.size() - 1;
  }

  return arrow::Status::OK();
}

arrow::Status ColumnGroups::add_metadatas(const std::vector<std::string_view>& keys,
                                          const std::vector<std::string_view>& values) {
  std::unordered_set<std::string_view> existing_keys;
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    if (existing_keys.find(keys[i]) != existing_keys.end()) {
      return arrow::Status::Invalid("Duplicate metadata key: " + std::string(keys[i]));
    }

    metadata_.emplace_back(keys[i], values[i]);
    existing_keys.insert(keys[i]);
  }
  return arrow::Status::OK();
}

}  // namespace milvus_storage::api
