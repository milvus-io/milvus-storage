#include "schema_util.h"
#include "parquet/exception.h"
#include "storage/options.h"
#include "common/exception.h"

std::shared_ptr<arrow::Schema>
BuildScalarSchema(arrow::Schema* schema, SpaceOption* options) {
  arrow::SchemaBuilder scalar_schema_builder;
  for (const auto& field : schema->fields()) {
    if (field->name() == options->vector_column) {
      continue;
    }
    PARQUET_THROW_NOT_OK(scalar_schema_builder.AddField(field));
  }
  PARQUET_ASSIGN_OR_THROW(auto scalar_schema, scalar_schema_builder.Finish());
  return scalar_schema;
}

std::shared_ptr<arrow::Schema>
BuildVectorSchema(arrow::Schema* schema, SpaceOption* options) {
  arrow::SchemaBuilder vector_schema_builder;
  for (const auto& field : schema->fields()) {
    if (field->name() == options->primary_column || field->name() == options->version_column ||
        field->name() == options->vector_column) {
      PARQUET_THROW_NOT_OK(vector_schema_builder.AddField(field));
    }
  }
  PARQUET_ASSIGN_OR_THROW(auto vector_schema, vector_schema_builder.Finish());
  return vector_schema;
}

std::shared_ptr<arrow::Schema>
BuildDeleteSchema(arrow::Schema* schema, SpaceOption* options) {
  arrow::SchemaBuilder delete_schema_builder;
  auto pk_field = schema->GetFieldByName(options->primary_column);
  auto version_field = schema->GetFieldByName(options->version_column);
  PARQUET_THROW_NOT_OK(delete_schema_builder.AddField(pk_field));
  PARQUET_THROW_NOT_OK(delete_schema_builder.AddField(version_field));
  PARQUET_ASSIGN_OR_THROW(auto delete_schema, delete_schema_builder.Finish());
  return delete_schema;
}

bool
ValidateSchema(arrow::Schema* schema, SpaceOption* options) {
  if (!schema->GetFieldByName(options->primary_column) ||
      !options->version_column.empty() && !schema->GetFieldByName(options->version_column)) {
    throw StorageException("version column not found");
  }
  return true;
}