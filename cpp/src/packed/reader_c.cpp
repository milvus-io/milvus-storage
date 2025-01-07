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
#include "common/log.h"
#include "packed/reader.h"
#include "filesystem/fs.h"
#include "common/config.h"

#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include <iostream>
#include <memory>

int Open(const char* path,
         struct ArrowSchema* schema,
         const int pk_index,
         const int ts_index,
         const int64_t buffer_size,
         struct ArrowArrayStream* out) {
  auto truePath = std::string(path);
  auto factory = std::make_shared<milvus_storage::FileSystemFactory>();
  auto conf = milvus_storage::StorageConfig();
  conf.uri = "file:///tmp/";
  auto r = factory->BuildFileSystem(conf, &truePath);
  if (!r.ok()) {
    LOG_STORAGE_ERROR_ << "Error building filesystem: " << path;
    return -2;
  }
  auto trueFs = r.value();
  auto trueSchema = arrow::ImportSchema(schema).ValueOrDie();
  auto reader = std::make_shared<milvus_storage::PackedRecordBatchReader>(*trueFs, path, trueSchema, pk_index, ts_index,
                                                                          buffer_size);
  auto status = ExportRecordBatchReader(reader, out);
  LOG_STORAGE_ERROR_ << "read export done";
  if (!status.ok()) {
    LOG_STORAGE_ERROR_ << "Error exporting record batch reader" << status.ToString();
    return static_cast<int>(status.code());
  }
  return 0;
}