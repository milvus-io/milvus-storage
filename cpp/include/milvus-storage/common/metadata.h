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

#pragma once

#include <parquet/metadata.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <sstream>
#include "arrow/table.h"
#include "milvus-storage/common/type_fwd.h"
#include "milvus-storage/packed/chunk_manager.h"

namespace milvus_storage {

class Metadata {
  public:
  virtual ~Metadata() = default;

  virtual std::string Serialize() const = 0;
  virtual void Deserialize(const std::string_view data) = 0;
};

class MetadataBuilder {
  private:
  struct MetadataHeader {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
  };

  static constexpr uint32_t kMagicNumber = 0x4D424C44;  // "MBLD" for Metadata BuiLD
  static constexpr uint32_t kCurrentVersion = 1;

  public:
  virtual ~MetadataBuilder() = default;
  MetadataBuilder() = default;

  void Append(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch) {
    metadata_collection_.emplace_back(Create(batch));
  }

  std::string Finish() { return MetadataBuilder::Serialize(metadata_collection_); }

  static std::string Serialize(const std::vector<std::unique_ptr<Metadata>>& metadata_list) {
    std::stringstream ss(std::ios::binary | std::ios::out);

    MetadataHeader header;
    header.magic = kMagicNumber;
    header.version = kCurrentVersion;
    header.count = metadata_list.size();
    ss.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto& meta : metadata_list) {
      std::string data = meta->Serialize();
      uint32_t len = data.length();
      ss.write(reinterpret_cast<const char*>(&len), sizeof(len));
      ss.write(data.data(), len);
    }
    return ss.str();
  }

  template <typename MetadataT>
  static std::vector<std::unique_ptr<MetadataT>> Deserialize(const std::string_view data) {
    if (data.empty()) {
      return {};
    }

    std::vector<std::unique_ptr<MetadataT>> result;

    size_t offset = 0;

    if (data.size() < sizeof(MetadataHeader)) {
      return {};
    }

    MetadataHeader header;
    std::memcpy(&header, data.data() + offset, sizeof(header));
    offset += sizeof(MetadataHeader);

    if (header.magic != kMagicNumber || header.version != kCurrentVersion || header.count == 0) {
      return {};
    }

    result.reserve(header.count);

    for (uint32_t i = 0; i < header.count; ++i) {
      if (offset + sizeof(uint32_t) > data.size()) {
        return {};
      }

      uint32_t len;
      std::memcpy(&len, data.data() + offset, sizeof(len));
      offset += sizeof(uint32_t);

      if (offset + len > data.size()) {
        return {};
      }

      std::string_view meta_data(data.data() + offset, len);
      offset += len;

      auto meta = std::make_unique<MetadataT>();
      meta->Deserialize(meta_data);
      result.emplace_back(std::move(meta));
    }

    return result;
  }

  protected:
  virtual std::unique_ptr<Metadata> Create(const std::vector<std::shared_ptr<arrow::RecordBatch>>& batch) = 0;
  std::vector<std::unique_ptr<Metadata>> metadata_collection_;
};

class FieldIDList {
  public:
  FieldIDList() = default;

  explicit FieldIDList(const std::vector<FieldID>& field_ids);

  bool operator==(const FieldIDList& other) const;

  void Add(FieldID field_id);

  FieldID Get(size_t index) const;

  size_t size() const;

  bool empty() const;

  static arrow::Result<FieldIDList> Make(const std::shared_ptr<arrow::Schema>& schema);

  std::string ToString() const;

  private:
  std::vector<FieldID> field_ids_;
};

class GroupFieldIDList {
  public:
  GroupFieldIDList() = default;

  explicit GroupFieldIDList(int64_t size);

  explicit GroupFieldIDList(const std::vector<std::vector<int>>& list);

  explicit GroupFieldIDList(const std::vector<FieldIDList>& list);

  static GroupFieldIDList Make(std::vector<std::vector<int>>& column_groups, FieldIDList& field_id_list);

  bool operator==(const GroupFieldIDList& other) const;

  void AddFieldIDList(const FieldIDList& field_ids);

  FieldIDList GetFieldIDList(size_t index) const;

  size_t num_groups() const;

  bool empty() const;

  std::string Serialize() const;

  static GroupFieldIDList Deserialize(const std::string& input);

  private:
  std::vector<FieldIDList> list_;
};

class RowGroupMetadata {
  public:
  RowGroupMetadata() = default;

  explicit RowGroupMetadata(size_t memory_size, int64_t row_num, int64_t row_offset);

  size_t memory_size() const;
  int64_t row_num() const;
  int64_t row_offset() const;

  std::string ToString() const;
  std::string Serialize() const;
  static RowGroupMetadata Deserialize(const std::string& input);

  private:
  size_t memory_size_;
  int64_t row_num_;
  int64_t row_offset_;
};

class RowGroupMetadataVector {
  public:
  RowGroupMetadataVector() = default;

  explicit RowGroupMetadataVector(const std::vector<RowGroupMetadata>& metadata);

  void Add(const RowGroupMetadata& metadata);

  const RowGroupMetadata& Get(size_t index) const;

  size_t size() const;

  size_t row_num() const;

  size_t memory_size() const;

  void clear();

  std::string ToString() const;

  std::string Serialize() const;

  static RowGroupMetadataVector Deserialize(const std::string& input);

  private:
  std::vector<RowGroupMetadata> vector_;
};

class PackedFileMetadata {
  public:
  PackedFileMetadata() = default;

  explicit PackedFileMetadata(const std::shared_ptr<parquet::FileMetaData>& metadata,
                              const RowGroupMetadataVector& row_group_metadata,
                              const std::map<FieldID, ColumnOffset>& field_id_mapping,
                              const GroupFieldIDList& group_field_id_list,
                              const std::string& storage_version);

  static arrow::Result<std::shared_ptr<PackedFileMetadata>> Make(std::shared_ptr<parquet::FileMetaData> metadata);

  const RowGroupMetadataVector GetRowGroupMetadataVector();

  const RowGroupMetadata& GetRowGroupMetadata(int index) const;

  const std::map<FieldID, ColumnOffset>& GetFieldIDMapping();

  const GroupFieldIDList GetGroupFieldIDList();

  const std::shared_ptr<parquet::FileMetaData>& GetParquetMetadata();

  template <typename MetadataT>
  std::vector<std::unique_ptr<MetadataT>> GetMetadataVector(std::string_view key) const {
    auto key_value_metadata = parquet_metadata_->key_value_metadata();
    auto metadata = key_value_metadata->Get(key);
    if (!metadata.ok()) {
      return {};
    }
    const std::string& metadata_str = metadata.ValueOrDie();
    return MetadataBuilder::Deserialize<MetadataT>(std::string_view(metadata_str));
  }

  const std::string& GetStorageVersion() const;

  int num_row_groups() const;

  size_t total_memory_size() const;

  private:
  std::shared_ptr<parquet::FileMetaData> parquet_metadata_;
  RowGroupMetadataVector row_group_metadata_;
  std::map<FieldID, ColumnOffset> field_id_mapping_;
  GroupFieldIDList group_field_id_list_;
  std::string storage_version_;
};

}  // namespace milvus_storage
