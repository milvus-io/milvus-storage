#pragma once

#include "filter/filter.h"
#include "proto/manifest.pb.h"

namespace milvus_storage {

struct WriteOption {
  int64_t max_record_per_file = 1024;
};

using FilterSet = std::vector<std::unique_ptr<Filter>>;
struct ReadOptions {
  FilterSet filters;
  std::vector<std::string> columns;  // must have pk and version
  int limit = -1;
  int version = -1;

  static ReadOptions& default_read_options() {
    static ReadOptions options;
    return options;
  }

  std::vector<std::string> output_columns() { return columns; }
  bool has_version() { return version != -1; }
};

struct SpaceOptions {
  std::string uri;

  bool operator==(const SpaceOptions& other) const { return uri == other.uri; }

  std::unique_ptr<manifest_proto::SpaceOptions> ToProtobuf();

  void FromProtobuf(const manifest_proto::SpaceOptions& options);
};

struct SchemaOptions {
  Status Validate(const arrow::Schema* schema);

  bool has_version_column() const { return !version_column.empty(); }

  std::unique_ptr<schema_proto::SchemaOptions> ToProtobuf();

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
