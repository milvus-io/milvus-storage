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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include "milvus-storage/manifest.h"
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"

namespace milvus_storage {
namespace api {

class DeleteEvaluator;

arrow::Result<std::shared_ptr<DeleteEvaluator>> CreateDeleteEvaluator(
    std::shared_ptr<Manifest> manifest,
    std::shared_ptr<arrow::Schema> schema,
    Properties properties,
    MaskedReadOptions options,
    std::function<std::string(const std::string&)> key_retriever);

const std::vector<std::string>& DeleteEvaluatorNeededColumns(const std::shared_ptr<DeleteEvaluator>& evaluator);

arrow::Result<std::shared_ptr<arrow::BooleanArray>> EvaluateDeleteKeepMask(
    const std::shared_ptr<DeleteEvaluator>& evaluator, const std::shared_ptr<arrow::RecordBatch>& batch);

}  // namespace api
}  // namespace milvus_storage
