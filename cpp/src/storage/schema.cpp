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

#include "milvus-storage/storage/schema.h"
#include <memory>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/common/utils.h"
#include "milvus-storage/common/log.h"
namespace milvus_storage {

Schema::Schema(std::shared_ptr<arrow::Schema> schema, SchemaOptions& options)
    : schema_(std::move(schema)), options_(options) {}

Status Schema::Validate() {
  RETURN_NOT_OK(options_.Validate(schema_.get()));
  RETURN_NOT_OK(BuildScalarSchema());
  RETURN_NOT_OK(BuildVectorSchema());
  RETURN_NOT_OK(BuildDeleteSchema());
  LOG_STORAGE_DEBUG_ << "Schema validate success";
  return Status::OK();
}

std::shared_ptr<arrow::Schema> Schema::schema() const { return schema_; }

const SchemaOptions& Schema::options() const { return options_; }

std::shared_ptr<arrow::Schema> Schema::scalar_schema() { return scalar_schema_; }

std::shared_ptr<arrow::Schema> Schema::vector_schema() { return vector_schema_; }

std::shared_ptr<arrow::Schema> Schema::delete_schema() { return delete_schema_; }

Result<std::unique_ptr<schema_proto::Schema>> Schema::ToProtobuf() {
  auto schema = std::make_unique<schema_proto::Schema>();
  ASSIGN_OR_RETURN_NOT_OK(auto arrow_schema, ToProtobufSchema(schema_.get()));

  auto options = options_.ToProtobuf();
  schema->set_allocated_arrow_schema(arrow_schema.release());
  schema->set_allocated_schema_options(options.release());
  return schema;
}

Status Schema::FromProtobuf(const schema_proto::Schema& schema) {
  ASSIGN_OR_RETURN_NOT_OK(schema_, FromProtobufSchema(schema.arrow_schema()));
  options_.FromProtobuf(schema.schema_options());
  RETURN_NOT_OK(BuildScalarSchema());
  RETURN_NOT_OK(BuildVectorSchema());
  RETURN_NOT_OK(BuildDeleteSchema());
  return Status::OK();
}

Status Schema::BuildScalarSchema() {
  arrow::SchemaBuilder scalar_schema_builder;
  for (const auto& field : schema_->fields()) {
    if (field->name() == options_.vector_column) {
      continue;
    }
    RETURN_ARROW_NOT_OK(scalar_schema_builder.AddField(field));
  }
  auto offset_field = std::make_shared<arrow::Field>(kOffsetFieldName, arrow::int64());
  RETURN_ARROW_NOT_OK(scalar_schema_builder.AddField(offset_field));
  ASSIGN_OR_RETURN_ARROW_NOT_OK(scalar_schema_, scalar_schema_builder.Finish());
  return Status::OK();
}

Status Schema::BuildVectorSchema() {
  arrow::SchemaBuilder vector_schema_builder;
  for (const auto& field : schema_->fields()) {
    if (field->name() == options_.primary_column || field->name() == options_.version_column ||
        field->name() == options_.vector_column) {
      RETURN_ARROW_NOT_OK(vector_schema_builder.AddField(field));
    }
  }
  ASSIGN_OR_RETURN_ARROW_NOT_OK(vector_schema_, vector_schema_builder.Finish());
  return Status::OK();
}

Status Schema::BuildDeleteSchema() {
  arrow::SchemaBuilder delete_schema_builder;
  auto pk_field = schema_->GetFieldByName(options_.primary_column);
  auto version_field = schema_->GetFieldByName(options_.version_column);
  RETURN_ARROW_NOT_OK(delete_schema_builder.AddField(pk_field));
  if (options_.has_version_column()) {
    RETURN_ARROW_NOT_OK(delete_schema_builder.AddField(version_field));
  }
  ASSIGN_OR_RETURN_ARROW_NOT_OK(delete_schema_, delete_schema_builder.Finish());
  return Status::OK();
}
}  // namespace milvus_storage
