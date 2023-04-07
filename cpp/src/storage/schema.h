#pragma once
#include <arrow/type.h>
#include <string>
#include "storage/options.h"
namespace milvus_storage {

class Schema {
  public:
  Schema() = default;
  Schema(std::shared_ptr<arrow::Schema> schema, const SchemaOptions& options);

  std::shared_ptr<arrow::Schema>
  schema();

  const SchemaOptions*
  options();

  std::shared_ptr<arrow::Schema>
  scalar_schema();

  std::shared_ptr<arrow::Schema>
  vector_schema();

  std::shared_ptr<arrow::Schema>
  delete_schema();

  std::unique_ptr<schema_proto::Schema>
  ToProtobuf();

  void
  FromProtobuf(const schema_proto::Schema& schema);

  private:
  void
  BuildScalarSchema();

  void
  BuildVectorSchema();

  void
  BuildDeleteSchema();

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schema_;
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::shared_ptr<arrow::Schema> delete_schema_;

  SchemaOptions options_;
};
}  // namespace milvus_storage