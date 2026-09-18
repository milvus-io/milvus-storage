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

#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Specific.hh>

#include "milvus-storage/table_format/types.h"

// Avro codec_traits specializations for table_format types.
// These must be visible to any translation unit that uses
// DataFileWriter<T> / DataFileReader<T> with these types.

namespace avro {

// ==================== Optional helpers ====================

inline void encodeOptionalString(Encoder& e, const std::optional<std::string>& val) {
  if (val.has_value()) {
    e.encodeUnionIndex(1);
    avro::encode(e, val.value());
  } else {
    e.encodeUnionIndex(0);
    e.encodeNull();
  }
}

inline void decodeOptionalString(Decoder& d, std::optional<std::string>& val) {
  auto idx = d.decodeUnionIndex();
  if (idx == 0) {
    d.decodeNull();
    val = std::nullopt;
  } else {
    std::string s;
    avro::decode(d, s);
    val = std::move(s);
  }
}

inline void encodeOptionalLong(Encoder& e, const std::optional<int64_t>& val) {
  if (val.has_value()) {
    e.encodeUnionIndex(1);
    avro::encode(e, val.value());
  } else {
    e.encodeUnionIndex(0);
    e.encodeNull();
  }
}

inline void decodeOptionalLong(Decoder& d, std::optional<int64_t>& val) {
  auto idx = d.decodeUnionIndex();
  if (idx == 0) {
    d.decodeNull();
    val = std::nullopt;
  } else {
    int64_t v;
    avro::decode(d, v);
    val = v;
  }
}

inline void encodeOptionalInt(Encoder& e, const std::optional<int32_t>& val) {
  if (val.has_value()) {
    e.encodeUnionIndex(1);
    avro::encode(e, val.value());
  } else {
    e.encodeUnionIndex(0);
    e.encodeNull();
  }
}

inline void decodeOptionalInt(Decoder& d, std::optional<int32_t>& val) {
  auto idx = d.decodeUnionIndex();
  if (idx == 0) {
    d.decodeNull();
    val = std::nullopt;
  } else {
    int32_t v;
    avro::decode(d, v);
    val = v;
  }
}

inline void encodeOptionalMap(Encoder& e, const std::optional<std::map<std::string, std::string>>& val) {
  if (val.has_value()) {
    e.encodeUnionIndex(1);
    avro::encode(e, val.value());
  } else {
    e.encodeUnionIndex(0);
    e.encodeNull();
  }
}

inline void decodeOptionalMap(Decoder& d, std::optional<std::map<std::string, std::string>>& val) {
  auto idx = d.decodeUnionIndex();
  if (idx == 0) {
    d.decodeNull();
    val = std::nullopt;
  } else {
    std::map<std::string, std::string> m;
    avro::decode(d, m);
    val = std::move(m);
  }
}

// ==================== Type codec_traits ====================

using namespace milvus_storage::api::table_format;

template <>
struct codec_traits<FieldSchema> {
  static void encode(Encoder& e, const FieldSchema& f) {
    avro::encode(e, f.field_id);
    avro::encode(e, f.name);
    avro::encode(e, static_cast<int32_t>(f.data_type));
    avro::encode(e, f.type_params);
    avro::encode(e, f.is_primary_key);
    avro::encode(e, f.is_partition_key);
    avro::encode(e, f.is_clustering_key);
    avro::encode(e, f.nullable);
    avro::encode(e, f.is_dynamic);
    avro::encode(e, f.is_function_output);
    std::optional<int32_t> elem_type;
    if (f.element_type.has_value()) {
      elem_type = static_cast<int32_t>(f.element_type.value());
    }
    encodeOptionalInt(e, elem_type);
    encodeOptionalString(e, f.default_value);
    encodeOptionalString(e, f.description);
    encodeOptionalString(e, f.external_field);
  }

  static void decode(Decoder& d, FieldSchema& f) {
    avro::decode(d, f.field_id);
    avro::decode(d, f.name);
    int32_t data_type_int;
    avro::decode(d, data_type_int);
    f.data_type = static_cast<DataType>(data_type_int);
    avro::decode(d, f.type_params);
    avro::decode(d, f.is_primary_key);
    avro::decode(d, f.is_partition_key);
    avro::decode(d, f.is_clustering_key);
    avro::decode(d, f.nullable);
    avro::decode(d, f.is_dynamic);
    avro::decode(d, f.is_function_output);
    std::optional<int32_t> elem_type_int;
    decodeOptionalInt(d, elem_type_int);
    if (elem_type_int.has_value()) {
      f.element_type = static_cast<DataType>(elem_type_int.value());
    }
    decodeOptionalString(d, f.default_value);
    decodeOptionalString(d, f.description);
    decodeOptionalString(d, f.external_field);
  }
};

template <>
struct codec_traits<FunctionSchema> {
  static void encode(Encoder& e, const FunctionSchema& f) {
    avro::encode(e, f.function_id);
    avro::encode(e, f.name);
    encodeOptionalString(e, f.description);
    avro::encode(e, f.type);
    avro::encode(e, f.input_field_names);
    avro::encode(e, f.input_field_ids);
    avro::encode(e, f.output_field_names);
    avro::encode(e, f.output_field_ids);
    avro::encode(e, f.params);
  }

  static void decode(Decoder& d, FunctionSchema& f) {
    avro::decode(d, f.function_id);
    avro::decode(d, f.name);
    decodeOptionalString(d, f.description);
    avro::decode(d, f.type);
    avro::decode(d, f.input_field_names);
    avro::decode(d, f.input_field_ids);
    avro::decode(d, f.output_field_names);
    avro::decode(d, f.output_field_ids);
    avro::decode(d, f.params);
  }
};

template <>
struct codec_traits<IndexInfo> {
  static void encode(Encoder& e, const IndexInfo& idx) {
    avro::encode(e, idx.index_id);
    avro::encode(e, idx.index_name);
    avro::encode(e, idx.field_id);
    avro::encode(e, idx.index_params);
    avro::encode(e, idx.type_params);
    avro::encode(e, idx.auto_index);
    encodeOptionalMap(e, idx.user_index_params);
    avro::encode(e, idx.created_at);
  }

  static void decode(Decoder& d, IndexInfo& idx) {
    avro::decode(d, idx.index_id);
    avro::decode(d, idx.index_name);
    avro::decode(d, idx.field_id);
    avro::decode(d, idx.index_params);
    avro::decode(d, idx.type_params);
    avro::decode(d, idx.auto_index);
    decodeOptionalMap(d, idx.user_index_params);
    avro::decode(d, idx.created_at);
  }
};

template <>
struct codec_traits<SchemaInfo> {
  static void encode(Encoder& e, const SchemaInfo& s) {
    avro::encode(e, s.schema_id);
    avro::encode(e, s.fields);
    avro::encode(e, s.functions);
  }

  static void decode(Decoder& d, SchemaInfo& s) {
    avro::decode(d, s.schema_id);
    avro::decode(d, s.fields);
    avro::decode(d, s.functions);
  }
};

template <>
struct codec_traits<IndexSpec> {
  static void encode(Encoder& e, const IndexSpec& s) {
    avro::encode(e, s.spec_id);
    avro::encode(e, s.indexes);
  }

  static void decode(Decoder& d, IndexSpec& s) {
    avro::decode(d, s.spec_id);
    avro::decode(d, s.indexes);
  }
};

template <>
struct codec_traits<CollectionInfo> {
  static void encode(Encoder& e, const CollectionInfo& c) {
    avro::encode(e, c.collection_id);
    avro::encode(e, c.name);
    avro::encode(e, c.db_id);
    avro::encode(e, c.created_at);
    avro::encode(e, c.properties);
  }

  static void decode(Decoder& d, CollectionInfo& c) {
    avro::decode(d, c.collection_id);
    avro::decode(d, c.name);
    avro::decode(d, c.db_id);
    avro::decode(d, c.created_at);
    avro::decode(d, c.properties);
  }
};

template <>
struct codec_traits<ManifestListInfo> {
  static void encode(Encoder& e, const ManifestListInfo& r) {
    avro::encode(e, r.manifest_list);
    avro::encode(e, r.partition_ids);
    avro::encode(e, r.partition_names);
  }

  static void decode(Decoder& d, ManifestListInfo& r) {
    avro::decode(d, r.manifest_list);
    avro::decode(d, r.partition_ids);
    avro::decode(d, r.partition_names);
  }
};

template <>
struct codec_traits<SnapshotEntry> {
  static void encode(Encoder& e, const SnapshotEntry& s) {
    avro::encode(e, s.snapshot_id);
    encodeOptionalLong(e, s.parent_snapshot_id);
    avro::encode(e, s.timestamp_ms);
    avro::encode(e, s.schema_id);
    avro::encode(e, s.index_spec_id);
    avro::encode(e, s.manifest_lists);
  }

  static void decode(Decoder& d, SnapshotEntry& s) {
    avro::decode(d, s.snapshot_id);
    decodeOptionalLong(d, s.parent_snapshot_id);
    avro::decode(d, s.timestamp_ms);
    avro::decode(d, s.schema_id);
    avro::decode(d, s.index_spec_id);
    avro::decode(d, s.manifest_lists);
  }
};

template <>
struct codec_traits<SegmentInfo> {
  static void encode(Encoder& e, const SegmentInfo& s) {
    avro::encode(e, s.segment_id);
    avro::encode(e, s.manifest);
    auto level_int = static_cast<int32_t>(s.level);
    avro::encode(e, level_int);
    avro::encode(e, s.num_rows);
    avro::encode(e, s.file_size);
    avro::encode(e, s.index_size);
    avro::encode(e, s.sorted);
    avro::encode(e, s.partition_key_sorted);
  }

  static void decode(Decoder& d, SegmentInfo& s) {
    avro::decode(d, s.segment_id);
    avro::decode(d, s.manifest);
    int32_t level_int;
    avro::decode(d, level_int);
    s.level = static_cast<SegmentLevel>(level_int);
    avro::decode(d, s.num_rows);
    avro::decode(d, s.file_size);
    avro::decode(d, s.index_size);
    avro::decode(d, s.sorted);
    avro::decode(d, s.partition_key_sorted);
  }
};

template <>
struct codec_traits<ManifestListEntry> {
  static void encode(Encoder& e, const ManifestListEntry& p) {
    avro::encode(e, p.partition_id);
    avro::encode(e, p.partition_name);
    avro::encode(e, p.segments);
  }

  static void decode(Decoder& d, ManifestListEntry& p) {
    avro::decode(d, p.partition_id);
    avro::decode(d, p.partition_name);
    avro::decode(d, p.segments);
  }
};

}  // namespace avro
