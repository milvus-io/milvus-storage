

#pragma once

#include <vector>
#include <memory>
#include "milvus-storage/filter/filter.h"
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
