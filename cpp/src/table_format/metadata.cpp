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

#include "milvus-storage/table_format/metadata.h"
#include "milvus-storage/table_format/types_codec.h"

#include <sstream>

#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Stream.hh>
#include <fmt/format.h>

namespace avro {

using CollMetadata = milvus_storage::api::table_format::Metadata;

template <>
struct codec_traits<CollMetadata> {
  static void encode(Encoder& e, const CollMetadata& m) {
    avro::encode(e, m.format_version_);
    avro::encode(e, m.collection_);
    avro::encode(e, m.schemas_);
    avro::encode(e, m.current_schema_id_);
    avro::encode(e, m.index_specs_);
    avro::encode(e, m.current_index_spec_id_);
    avro::encode(e, m.snapshots_);
    avro::encode(e, m.current_snapshot_id_);
    avro::encode(e, m.next_snapshot_id_);
  }

  static void decode(Decoder& d, CollMetadata& m) {
    avro::decode(d, m.format_version_);
    avro::decode(d, m.collection_);
    avro::decode(d, m.schemas_);
    avro::decode(d, m.current_schema_id_);
    avro::decode(d, m.index_specs_);
    avro::decode(d, m.current_index_spec_id_);
    avro::decode(d, m.snapshots_);
    avro::decode(d, m.current_snapshot_id_);
    avro::decode(d, m.next_snapshot_id_);
    // Repair: ensure next_snapshot_id is greater than all existing snapshot IDs
    // (handles backward compatibility when deserializing old metadata without this field)
    for (const auto& snap : m.snapshots_) {
      if (snap.snapshot_id >= m.next_snapshot_id_) {
        m.next_snapshot_id_ = snap.snapshot_id + 1;
      }
    }
  }
};

}  // namespace avro

namespace milvus_storage::api::table_format {

static const char* const METADATA_SCHEMA_JSON = R"({
  "type": "record",
  "name": "Metadata",
  "namespace": "milvus_storage.table_format",
  "fields": [
    {"name": "format_version", "type": "int", "default": 1},
    {"name": "collection", "type": {
      "type": "record", "name": "CollectionInfo", "fields": [
        {"name": "collection_id", "type": "long"},
        {"name": "name", "type": "string"},
        {"name": "db_id", "type": "long"},
        {"name": "created_at", "type": "long"},
        {"name": "properties", "type": {"type": "map", "values": "string"}, "default": {}}
      ]
    }},
    {"name": "schemas", "type": {"type": "array", "items": {
      "type": "record", "name": "SchemaInfo", "fields": [
        {"name": "schema_id", "type": "int"},
        {"name": "fields", "type": {"type": "array", "items": {
          "type": "record", "name": "FieldSchema", "fields": [
            {"name": "field_id", "type": "long"},
            {"name": "name", "type": "string"},
            {"name": "data_type", "type": "int"},
            {"name": "type_params", "type": {"type": "map", "values": "string"}, "default": {}},
            {"name": "is_primary_key", "type": "boolean", "default": false},
            {"name": "is_partition_key", "type": "boolean", "default": false},
            {"name": "is_clustering_key", "type": "boolean", "default": false},
            {"name": "nullable", "type": "boolean", "default": false},
            {"name": "is_dynamic", "type": "boolean", "default": false},
            {"name": "is_function_output", "type": "boolean", "default": false},
            {"name": "element_type", "type": ["null", "int"], "default": null},
            {"name": "default_value", "type": ["null", "string"], "default": null},
            {"name": "description", "type": ["null", "string"], "default": null},
            {"name": "external_field", "type": ["null", "string"], "default": null}
          ]
        }}},
        {"name": "functions", "type": {"type": "array", "items": {
          "type": "record", "name": "FunctionSchema", "fields": [
            {"name": "function_id", "type": "long"},
            {"name": "name", "type": "string"},
            {"name": "description", "type": ["null", "string"], "default": null},
            {"name": "type", "type": "string"},
            {"name": "input_field_names", "type": {"type": "array", "items": "string"}},
            {"name": "input_field_ids", "type": {"type": "array", "items": "long"}},
            {"name": "output_field_names", "type": {"type": "array", "items": "string"}},
            {"name": "output_field_ids", "type": {"type": "array", "items": "long"}},
            {"name": "params", "type": {"type": "map", "values": "string"}, "default": {}}
          ]
        }}, "default": []}
      ]
    }}},
    {"name": "current_schema_id", "type": "int", "default": 0},
    {"name": "index_specs", "type": {"type": "array", "items": {
      "type": "record", "name": "IndexSpec", "fields": [
        {"name": "spec_id", "type": "int"},
        {"name": "indexes", "type": {"type": "array", "items": {
          "type": "record", "name": "IndexInfo", "fields": [
            {"name": "index_id", "type": "long"},
            {"name": "index_name", "type": "string"},
            {"name": "field_id", "type": "long"},
            {"name": "index_params", "type": {"type": "map", "values": "string"}, "default": {}},
            {"name": "type_params", "type": {"type": "map", "values": "string"}, "default": {}},
            {"name": "auto_index", "type": "boolean", "default": false},
            {"name": "user_index_params", "type": ["null", {"type": "map", "values": "string"}], "default": null},
            {"name": "created_at", "type": "long"}
          ]
        }}}
      ]
    }}, "default": []},
    {"name": "current_index_spec_id", "type": "int", "default": 0},
    {"name": "snapshots", "type": {"type": "array", "items": {
      "type": "record", "name": "SnapshotEntry", "fields": [
        {"name": "snapshot_id", "type": "long"},
        {"name": "parent_snapshot_id", "type": ["null", "long"], "default": null},
        {"name": "timestamp_ms", "type": "long"},
        {"name": "schema_id", "type": "int"},
        {"name": "index_spec_id", "type": "int"},
        {"name": "manifest_lists", "type": {"type": "array", "items": {
          "type": "record", "name": "ManifestListInfo", "fields": [
            {"name": "manifest_list", "type": "string"},
            {"name": "partition_ids", "type": {"type": "array", "items": "long"}, "default": []},
            {"name": "partition_names", "type": {"type": "array", "items": "string"}, "default": []}
          ]
        }}}
      ]
    }}, "default": []},
    {"name": "current_snapshot_id", "type": "long", "default": 0},
    {"name": "next_snapshot_id", "type": "long", "default": 1}
  ]
})";

static const avro::ValidSchema& getMetadataSchema() {
  static const avro::ValidSchema schema = avro::compileJsonSchemaFromString(METADATA_SCHEMA_JSON);
  return schema;
}

arrow::Status Metadata::serialize(std::ostream& output_stream) const {
  try {
    auto avro_output = avro::ostreamOutputStream(output_stream);
    avro::DataFileWriter<Metadata> writer(std::move(avro_output), getMetadataSchema());
    writer.write(*this);
    writer.close();
    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to serialize Metadata: {}", e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to serialize Metadata: {}", e.what()));
  }
}

arrow::Status Metadata::deserialize(std::istream& input_stream) {
  try {
    auto avro_input = avro::istreamInputStream(input_stream);
    avro::DataFileReader<Metadata> reader(std::move(avro_input), getMetadataSchema());
    if (!reader.read(*this)) {
      return arrow::Status::Invalid("Failed to deserialize Metadata: no record in Avro file");
    }
    return arrow::Status::OK();
  } catch (const avro::Exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to deserialize Metadata: {}", e.what()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Failed to deserialize Metadata: {}", e.what()));
  }
}

}  // namespace milvus_storage::api::table_format
