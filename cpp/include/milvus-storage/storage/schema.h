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

#pragma once
#include "options.h"
#include "common/result.h"
namespace milvus_storage {

class Schema {
  public:
  Schema() = default;
  Schema(std::shared_ptr<arrow::Schema> schema, SchemaOptions& options);

  Status Validate();

  std::shared_ptr<arrow::Schema> schema() const;

  const SchemaOptions& options() const;

  std::shared_ptr<arrow::Schema> scalar_schema();

  std::shared_ptr<arrow::Schema> vector_schema();

  std::shared_ptr<arrow::Schema> delete_schema();

  Result<std::unique_ptr<schema_proto::Schema>> ToProtobuf();

  Status FromProtobuf(const schema_proto::Schema& schema);

  private:
  Status BuildScalarSchema();

  Status BuildVectorSchema();

  Status BuildDeleteSchema();

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> scalar_schema_;
  std::shared_ptr<arrow::Schema> vector_schema_;
  std::shared_ptr<arrow::Schema> delete_schema_;

  SchemaOptions options_;
};
}  // namespace milvus_storage
