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
#include <memory>
#include "parquet/arrow/reader.h"
#include "arrow/filesystem/filesystem.h"
#include "common/result.h"
#include "storage/options.h"

namespace milvus_storage {
Result<std::shared_ptr<parquet::arrow::FileReader>> MakeArrowFileReader(arrow::fs::FileSystem& fs,
                                                                        const std::string& file_path);

Result<std::shared_ptr<arrow::RecordBatchReader>> MakeArrowRecordBatchReader(
    std::shared_ptr<parquet::arrow::FileReader> reader,
    const ReadOptions& options = ReadOptions());
}  // namespace milvus_storage
