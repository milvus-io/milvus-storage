#pragma once
#include <arrow/type.h>
#include <set>
#include <string>
#include <vector>

#include "../filter/filter.h"
#include "proto/manifest.pb.h"
#include <filesystem>
struct WriteOption {
  int64_t max_record_per_file = 1024;
};

struct ReadOptions {
  std::vector<Filter*> filters;
  std::vector<std::string> columns;  // must have pk and version
  int limit = -1;
  int version = -1;
};

struct SpaceOptions {
  std::string uri;

  std::unique_ptr<manifest_proto::SpaceOptions>
  ToProtobuf();

  void
  FromProtobuf(const manifest_proto::SpaceOptions& options);
};

struct SchemaOptions {
  std::string primary_column;  // must not  null, int64 or string
  // version column is to support mvcc and it can be set in ReadOption.
  // if it's not empty, the column type must be int64
  std::string version_column = "";
  std::string vector_column = "";  // could be null, fixed length binary

  bool
  Validate(const arrow::Schema* schema);

  std::unique_ptr<schema_proto::SchemaOptions>
  ToProtobuf();

  void
  FromProtobuf(const schema_proto::SchemaOptions& options);
};