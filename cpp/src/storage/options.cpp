

#include "milvus-storage/storage/options.h"
#include <arrow/type_fwd.h>
#include "arrow/type.h"
#include "milvus-storage/common/status.h"

namespace milvus_storage {

Status SchemaOptions::Validate(const arrow::Schema* schema) const {
  if (!primary_column.empty()) {
    auto primary_field = schema->GetFieldByName(primary_column);
    if (!primary_field) {
      return Status::InvalidArgument("primary column is not exist");
    } else if (primary_field->type()->id() != arrow::Type::INT64 &&
               primary_field->type()->id() != arrow::Type::STRING) {
      return Status::InvalidArgument("primary column is not int64 or string");
    }
  } else {
    return Status::InvalidArgument("primary column is empty");
  }

  if (!version_column.empty()) {
    auto version_field = schema->GetFieldByName(version_column);
    if (!version_field) {
      return Status::InvalidArgument("version column is not exist");
    } else if (version_field->type()->id() != arrow::Type::INT64) {
      return Status::InvalidArgument("version column is not int64");
    }
  }

  if (!vector_column.empty()) {
    auto vector_field = schema->GetFieldByName(vector_column);
    if (!vector_field) {
      return Status::InvalidArgument("vector column is not exist");
    } else if (vector_field->type()->id() != arrow::Type::FIXED_SIZE_BINARY &&
               vector_field->type()->id() != arrow::Type::FIXED_SIZE_LIST) {
      return Status::InvalidArgument("vector column is not fixed size binary or fixed size list");
    }
  } else {
    return Status::InvalidArgument("vector column is empty");
  }

  return Status::OK();
}

std::unique_ptr<schema_proto::SchemaOptions> SchemaOptions::ToProtobuf() const {
  auto options = std::make_unique<schema_proto::SchemaOptions>();
  options->set_primary_column(primary_column);
  options->set_version_column(version_column);
  options->set_vector_column(vector_column);
  return options;
}

void SchemaOptions::FromProtobuf(const schema_proto::SchemaOptions& options) {
  primary_column = options.primary_column();
  version_column = options.version_column();
  vector_column = options.vector_column();
}
}  // namespace milvus_storage
