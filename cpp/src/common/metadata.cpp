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

#include <cstdint>
#include <memory>
#include <string>
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/result.h"
#include "milvus-storage/common/status.h"
#include "milvus-storage/common/type_fwd.h"
#include <sstream>
#include <iostream>

namespace milvus_storage {

// RowGroupSizeVector implementation
RowGroupSizeVector::RowGroupSizeVector(const std::vector<size_t>& size) : vector_(size) {}

void RowGroupSizeVector::Add(size_t size) { vector_.push_back(size); }

size_t RowGroupSizeVector::Get(size_t index) const {
  if (index >= vector_.size()) {
    throw std::out_of_range("Get row group size failed: out of range size " + std::to_string(index));
  }
  return vector_[index];
}

size_t RowGroupSizeVector::size() const { return vector_.size(); }

size_t RowGroupSizeVector::memory_size() const {
  size_t size = 0;
  for (size_t i = 0; i < vector_.size(); ++i) {
    size += vector_[i];
  }
  return size;
}

void RowGroupSizeVector::clear() { vector_.clear(); }

std::string RowGroupSizeVector::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < vector_.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << vector_[i];
  }
  return ss.str();
}

std::string RowGroupSizeVector::Serialize() const {
  std::vector<uint8_t> byteArray(vector_.size() * sizeof(size_t));
  std::memcpy(byteArray.data(), vector_.data(), byteArray.size());
  return std::string(byteArray.begin(), byteArray.end());
}

RowGroupSizeVector RowGroupSizeVector::Deserialize(const std::string& input) {
  std::vector<uint8_t> byteArray(input.begin(), input.end());
  std::vector<size_t> sizes(byteArray.size() / sizeof(size_t));
  std::memcpy(sizes.data(), byteArray.data(), byteArray.size());
  return RowGroupSizeVector(sizes);
}

// Implementation of FieldIDList
FieldIDList::FieldIDList(const std::vector<FieldID>& field_ids) : field_ids_(std::move(field_ids)) {}

bool FieldIDList::operator==(const FieldIDList& other) const { return field_ids_ == other.field_ids_; }

void FieldIDList::Add(FieldID field_id) { field_ids_.push_back(field_id); }

FieldID FieldIDList::Get(size_t index) const {
  if (index >= field_ids_.size()) {
    throw std::out_of_range("Get field id failed: out of range size " + std::to_string(index));
  }
  return field_ids_[index];
}

size_t FieldIDList::size() const { return field_ids_.size(); }

bool FieldIDList::empty() const { return field_ids_.empty(); }

Result<FieldIDList> FieldIDList::Make(const std::shared_ptr<arrow::Schema>& schema) {
  FieldIDList field_ids;
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto metadata = schema->field(i)->metadata();
    if (!metadata || !metadata->Contains(ARROW_FIELD_ID_KEY)) {
      return Status::InvalidArgument("field metadata is null");
    }
    auto field = metadata->Get(ARROW_FIELD_ID_KEY).ValueOrDie();
    field_ids.Add(std::stoll(field));
  }
  return field_ids;
}

std::string FieldIDList::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < field_ids_.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << field_ids_[i];
  }
  return ss.str();
}

// Implementation of GroupFieldIDList
GroupFieldIDList::GroupFieldIDList(int64_t size) : list_(size) {}

GroupFieldIDList::GroupFieldIDList(const std::vector<std::vector<int>>& list) {
  for (const auto& group : list) {
    FieldIDList field_ids;
    for (int i : group) {
      field_ids.Add(i);
    }
    list_.push_back(field_ids);
  }
}

GroupFieldIDList::GroupFieldIDList(const std::vector<FieldIDList>& list) : list_(std::move(list)) {}

GroupFieldIDList GroupFieldIDList::Make(std::vector<std::vector<int>>& column_groups, FieldIDList& field_id_list) {
  GroupFieldIDList list;
  for (const auto& group_index : column_groups) {
    FieldIDList field_ids;
    for (int i : group_index) {
      field_ids.Add(field_id_list.Get(i));
    }
    list.AddFieldIDList(field_ids);
  }
  return list;
}

bool GroupFieldIDList::operator==(const GroupFieldIDList& other) const { return list_ == other.list_; }

void GroupFieldIDList::AddFieldIDList(const FieldIDList& field_ids) { list_.push_back(field_ids); }

FieldIDList GroupFieldIDList::GetFieldIDList(size_t index) const {
  if (index >= list_.size()) {
    throw std::out_of_range("Get field id list failed: out of range size " + std::to_string(index));
  }
  return list_[index];
}

size_t GroupFieldIDList::num_groups() const { return list_.size(); }

bool GroupFieldIDList::empty() const { return list_.empty(); }

std::string GroupFieldIDList::Serialize() const {
  std::stringstream ss;
  for (size_t i = 0; i < list_.size(); ++i) {
    if (i > 0) {
      ss << GROUP_DELIMITER;
    }
    for (size_t j = 0; j < list_[i].size(); ++j) {
      if (j > 0) {
        ss << COLUMN_DELIMITER;
      }
      ss << list_[i].Get(j);
    }
  }
  return ss.str();
}

GroupFieldIDList GroupFieldIDList::Deserialize(const std::string& input) {
  std::vector<FieldIDList> group_field_id_list;
  size_t group_start = 0;
  size_t group_end = input.find(GROUP_DELIMITER);
  while (group_start != std::string::npos) {
    std::string group = input.substr(group_start, group_end - group_start);
    FieldIDList field_id_list;
    size_t column_start = 0;
    size_t column_end = group.find(COLUMN_DELIMITER);
    while (column_start != std::string::npos) {
      std::string field_id = group.substr(column_start, column_end - column_start);
      if (!field_id.empty()) {
        field_id_list.Add(std::stoll(field_id));
      }
      column_start = (column_end == std::string::npos) ? std::string::npos : column_end + COLUMN_DELIMITER.size();
      column_end = group.find(COLUMN_DELIMITER, column_start);
    }
    if (!field_id_list.empty()) {
      group_field_id_list.push_back(field_id_list);
    }
    group_start = (group_end == std::string::npos) ? std::string::npos : group_end + GROUP_DELIMITER.size();
    group_end = input.find(GROUP_DELIMITER, group_start);
  }
  return GroupFieldIDList(group_field_id_list);
}

// Implementation of PackedFileMetadata

PackedFileMetadata::PackedFileMetadata(const std::shared_ptr<parquet::FileMetaData>& metadata,
                                       const RowGroupSizeVector& row_group_sizes,
                                       const std::map<FieldID, ColumnOffset>& field_id_mapping,
                                       const GroupFieldIDList& group_field_id_list)
    : parquet_metadata_(std::move(metadata)),
      row_group_sizes_(std::move(row_group_sizes)),
      field_id_mapping_(std::move(field_id_mapping)),
      group_field_id_list_(std::move(group_field_id_list)) {}

Result<std::shared_ptr<PackedFileMetadata>> PackedFileMetadata::Make(std::shared_ptr<parquet::FileMetaData> metadata) {
  // deserialize row group size metadata
  auto key_value_metadata = metadata->key_value_metadata();
  auto row_group_size_meta = key_value_metadata->Get(ROW_GROUP_SIZE_META_KEY);
  if (!row_group_size_meta.ok()) {
    return Status::MetadataParseError("Row group size metadata not found");
  }
  auto row_group_sizes = RowGroupSizeVector::Deserialize(row_group_size_meta.ValueOrDie());

  // deserialize field id mapping metadata
  auto group_field_id_list_meta = key_value_metadata->Get(GROUP_FIELD_ID_LIST_META_KEY);
  if (!group_field_id_list_meta.ok()) {
    return Status::MetadataParseError("Field id list metadata not found");
  }
  auto group_fields = GroupFieldIDList::Deserialize(group_field_id_list_meta.ValueOrDie());
  std::map<FieldID, ColumnOffset> field_id_mapping;
  for (int path = 0; path < group_fields.num_groups(); path++) {
    auto field_ids = group_fields.GetFieldIDList(path);
    for (int col = 0; col < field_ids.size(); col++) {
      FieldID field_id = field_ids.Get(col);
      field_id_mapping[field_id] = ColumnOffset(path, col);
    }
  }
  return std::make_shared<PackedFileMetadata>(metadata, row_group_sizes, field_id_mapping, group_fields);
}

const RowGroupSizeVector PackedFileMetadata::GetRowGroupSizeVector() { return row_group_sizes_; }

size_t PackedFileMetadata::GetRowGroupSize(int index) const { return row_group_sizes_.Get(index); }

const std::map<FieldID, ColumnOffset>& PackedFileMetadata::GetFieldIDMapping() { return field_id_mapping_; }

const GroupFieldIDList PackedFileMetadata::GetGroupFieldIDList() { return group_field_id_list_; }

const std::shared_ptr<parquet::FileMetaData>& PackedFileMetadata::GetParquetMetadata() { return parquet_metadata_; }

int PackedFileMetadata::num_row_groups() const { return row_group_sizes_.size(); }

size_t PackedFileMetadata::total_memory_size() const { return row_group_sizes_.memory_size(); }

}  // namespace milvus_storage
