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
#include "arrow/record_batch.h"
#include "common/status.h"
namespace milvus_storage {

class FileWriter {
  public:
  virtual Status Init() = 0;

  virtual Status Write(const arrow::RecordBatch& record) = 0;

  virtual Status WriteTable(const arrow::Table& table) = 0;

  virtual int64_t count() = 0;

  virtual Status Close() = 0;

  virtual ~FileWriter() = default;
};
}  // namespace milvus_storage
