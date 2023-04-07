#include "storage/options.h"
namespace milvus_storage {

std::unique_ptr<manifest_proto::SpaceOptions>
SpaceOptions::ToProtobuf() {
  auto options = std::make_unique<manifest_proto::SpaceOptions>();
  options->set_uri(uri);
  return options;
}

void
SpaceOptions::FromProtobuf(const manifest_proto::SpaceOptions& options) {
  uri = options.uri();
}

bool
SchemaOptions::Validate(const arrow::Schema* schema) {
  if (primary_column.empty() || !schema->GetFieldByName(primary_column)) {
    return false;
  }

  if (!version_column.empty()) {
    auto version_field = schema->GetFieldByName(version_column);
    if (!version_field || version_field->type()->id() != arrow::Type::INT64) {
      return false;
    }
  }

  if (!vector_column.empty()) {
    auto vector_field = schema->GetFieldByName(vector_column);
    if (!vector_field || vector_field->type()->id() != arrow::Type::FIXED_SIZE_BINARY) {
      return false;
    }
  }

  return true;
}

std::unique_ptr<schema_proto::SchemaOptions>
SchemaOptions::ToProtobuf() {
  auto options = std::make_unique<schema_proto::SchemaOptions>();
  options->set_primary_column(primary_column);
  options->set_version_column(version_column);
  options->set_vector_column(vector_column);
  return options;
}

void
SchemaOptions::FromProtobuf(const schema_proto::SchemaOptions& options) {
  primary_column = options.primary_column();
  version_column = options.version_column();
  vector_column = options.vector_column();
}
}  // namespace milvus_storage