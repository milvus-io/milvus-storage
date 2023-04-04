#pragma once
#include <arrow/type.h>
#include <string>

struct SchemaOptions {
  std::string primary_column;  // must not  null, int64 or string
  std::string version_column;  // could be null, int64
  std::string vector_column;   // could be null, fixed length binary
};

class Schema {
  public:
  Schema(std::shared_ptr<arrow::Schema> schema, const SchemaOptions& options);

  std::shared_ptr<arrow::Schema>
  GetSchema();

  const SchemaOptions*
  GetOptions();

  std::shared_ptr<arrow::Schema>
  GetScalarSchema();

  std::shared_ptr<arrow::Schema>
  GetVectorSchema();

  std::shared_ptr<arrow::Schema>
  GetDeleteSchema();

  private:
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schema_;
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::shared_ptr<arrow::Schema> delete_schema_;
  SchemaOptions options_;
};