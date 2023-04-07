#include "schema.h"

#include <utility>
#include "parquet/exception.h"
#include "common/exception.h"
#include "common/utils.h"
namespace milvus_storage {

Schema::Schema(std::shared_ptr<arrow::Schema> schema, const SchemaOptions& options)
    : schema_(std::move(schema)), options_(options) {
  if (!options_.Validate(schema_.get())) {
    throw StorageException("invalid schema");
  }
  BuildScalarSchema();
  BuildVectorSchema();
  BuildDeleteSchema();
}

std::shared_ptr<arrow::Schema>
Schema::schema() {
  return schema_;
}

const SchemaOptions*
Schema::options() {
  return &options_;
}

std::shared_ptr<arrow::Schema>
Schema::scalar_schema() {
  return scalar_schema_;
}

std::shared_ptr<arrow::Schema>
Schema::vector_schema() {
  return vector_schema_;
}

std::shared_ptr<arrow::Schema>
Schema::delete_schema() {
  return delete_schema_;
}

std::unique_ptr<schema_proto::Schema>
Schema::ToProtobuf() {
  auto schema = std::make_unique<schema_proto::Schema>();
  auto arrow_schema = ToProtobufSchema(schema_.get());
  auto options = options_.ToProtobuf();
  schema->set_allocated_arrow_schema(arrow_schema.release());
  schema->set_allocated_schema_options(options.release());
  return schema;
}

void
Schema::FromProtobuf(const schema_proto::Schema& schema) {
  schema_ = FromProtobufSchema(schema.arrow_schema());
  options_.FromProtobuf(schema.schema_options());
  BuildScalarSchema();
  BuildVectorSchema();
  BuildDeleteSchema();
}

void
Schema::BuildScalarSchema() {
  arrow::SchemaBuilder scalar_schema_builder;
  for (const auto& field : schema_->fields()) {
    if (field->name() == options_.vector_column) {
      continue;
    }
    PARQUET_THROW_NOT_OK(scalar_schema_builder.AddField(field));
  }
  PARQUET_ASSIGN_OR_THROW(scalar_schema_, scalar_schema_builder.Finish());
}

void
Schema::BuildVectorSchema() {
  arrow::SchemaBuilder vector_schema_builder;
  for (const auto& field : schema_->fields()) {
    if (field->name() == options_.primary_column || field->name() == options_.version_column ||
        field->name() == options_.vector_column) {
      PARQUET_THROW_NOT_OK(vector_schema_builder.AddField(field));
    }
  }
  PARQUET_ASSIGN_OR_THROW(vector_schema_, vector_schema_builder.Finish());
}

void
Schema::BuildDeleteSchema() {
  arrow::SchemaBuilder delete_schema_builder;
  auto pk_field = schema_->GetFieldByName(options_.primary_column);
  auto version_field = schema_->GetFieldByName(options_.version_column);
  PARQUET_THROW_NOT_OK(delete_schema_builder.AddField(pk_field));
  PARQUET_THROW_NOT_OK(delete_schema_builder.AddField(version_field));
  PARQUET_ASSIGN_OR_THROW(delete_schema_, delete_schema_builder.Finish());
}
}  // namespace milvus_storage