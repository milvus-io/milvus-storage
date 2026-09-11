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

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace milvus_storage::api::table_format {

enum class DataType : int32_t {
  None = 0,
  Bool = 1,
  Int8 = 2,
  Int16 = 3,
  Int32 = 4,
  Int64 = 5,
  Float = 10,
  Double = 11,
  String = 20,
  VarChar = 21,
  Array = 22,
  JSON = 23,
  Geometry = 24,
  Text = 25,
  Timestamptz = 26,
  Mol = 27,
  BinaryVector = 100,
  FloatVector = 101,
  Float16Vector = 102,
  BFloat16Vector = 103,
  SparseFloatVector = 104,
  Int8Vector = 105,
  ArrayOfVector = 106,
  ArrayOfStruct = 200,
  Struct = 201,
};

struct FieldSchema {
  int64_t field_id = 0;
  std::string name;
  DataType data_type = DataType::None;
  std::map<std::string, std::string> type_params;
  bool is_primary_key = false;
  bool is_partition_key = false;
  bool is_clustering_key = false;
  bool nullable = false;
  bool is_dynamic = false;
  bool is_function_output = false;
  std::optional<DataType> element_type;
  std::optional<std::string> default_value;
  std::optional<std::string> description;
  std::optional<std::string> external_field;
};

struct FunctionSchema {
  int64_t function_id = 0;
  std::string name;
  std::optional<std::string> description;
  std::string type;
  std::vector<std::string> input_field_names;
  std::vector<int64_t> input_field_ids;
  std::vector<std::string> output_field_names;
  std::vector<int64_t> output_field_ids;
  std::map<std::string, std::string> params;
};

struct IndexInfo {
  int64_t index_id = 0;
  std::string index_name;
  int64_t field_id = 0;
  std::map<std::string, std::string> index_params;
  std::map<std::string, std::string> type_params;
  bool auto_index = false;
  std::optional<std::map<std::string, std::string>> user_index_params;
  int64_t created_at = 0;
};

struct SchemaInfo {
  int32_t schema_id = 0;
  std::vector<FieldSchema> fields;
  std::vector<FunctionSchema> functions;
};

struct IndexSpec {
  int32_t spec_id = 0;
  std::vector<IndexInfo> indexes;
};

struct CollectionInfo {
  int64_t collection_id = 0;
  std::string name;
  int64_t db_id = 0;
  int64_t created_at = 0;
  std::map<std::string, std::string> properties;
};

struct ManifestListInfo {
  std::string manifest_list;
  std::vector<int64_t> partition_ids;
  std::vector<std::string> partition_names;
};

struct SnapshotEntry {
  int64_t snapshot_id = 0;
  std::optional<int64_t> parent_snapshot_id;
  int64_t timestamp_ms = 0;
  int32_t schema_id = 0;
  int32_t index_spec_id = 0;
  std::vector<ManifestListInfo> manifest_lists;
};

enum class SegmentLevel { L1, L2 };

struct SegmentInfo {
  int64_t segment_id = 0;
  std::string manifest;
  SegmentLevel level = SegmentLevel::L1;
  int64_t num_rows = 0;
  int64_t file_size = 0;
  int64_t index_size = 0;
  bool sorted = false;
  bool partition_key_sorted = false;
};

struct ManifestListEntry {
  int64_t partition_id = 0;
  std::string partition_name;
  std::vector<SegmentInfo> segments;
};

}  // namespace milvus_storage::api::table_format
