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

#include "packed/reader_c.h"
#include "packed/reader.h"
#include "filesystem/fs.h"

#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include <memory>

int Open(const char* path, struct ArrowSchema* schema, const int64_t buffer_size, struct ArrowArrayStream* out) {

    auto truePath = std::string(path);
    auto factory = std::make_shared<milvus_storage::FileSystemFactory>();
    auto r = factory->BuildFileSystem(truePath, &truePath);
    if (!r.ok()) {
        return -1;
    }
    auto trueFs = r.value();
    auto trueSchema = arrow::ImportSchema(schema).ValueOrDie();
    auto reader = std::make_shared<milvus_storage::PackedRecordBatchReader>(*trueFs, path, trueSchema, buffer_size);
    auto status = ExportRecordBatchReader(reader, out);
    if (!status.ok()) {
        return -1;
    }
    return 0;
}