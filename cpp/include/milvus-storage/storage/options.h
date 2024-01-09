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

#pragma once

#include <vector>
#include <memory>
#include "filter/filter.h"
#include "proto/manifest.pb.h"

namespace milvus_storage {

class Schema;
struct Options {
  std::shared_ptr<Schema> schema = nullptr;
  int64_t version = -1;
};

struct WriteOption {
  int64_t max_record_per_file = 1024;
};

struct ReadOptions {
  Filter::FilterSet filters;

  std::set<std::string> columns;  // empty means all columns
  // int limit = -1;
  int64_t version = INT64_MAX;

  static bool ReturnAllColumns(const ReadOptions& options) { return options.columns.empty(); }
};

struct SchemaOptions {
  Status Validate(const arrow::Schema* schema) const;

  bool has_version_column() const { return !version_column.empty(); }

  std::unique_ptr<schema_proto::SchemaOptions> ToProtobuf() const;

  void FromProtobuf(const schema_proto::SchemaOptions& options);

  bool operator==(const SchemaOptions& other) const {
    return primary_column == other.primary_column && version_column == other.version_column &&
           vector_column == other.vector_column;
  }

  // primary_column must not be empty and the type must be int64 or string
  std::string primary_column;
  // version_column is to support mvcc and it can be set in ReadOption.
  // it can be empty and if not, the column type must be int64
  std::string version_column;
  // vector_column must not be emtpy, and the type must be fixed size binary
  std::string vector_column;
};

}  // namespace milvus_storage
