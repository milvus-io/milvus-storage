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

#include "packed/writer_c.h"
#include "packed/writer.h"
#include "common/log.h"
#include "common/config.h"
#include "filesystem/fs.h"

#include <arrow/c/bridge.h>
#include <arrow/filesystem/filesystem.h>
#include <iostream>

int NewPackedWriter(const char* path,
                    struct ArrowSchema* schema,
                    const int64_t buffer_size,
                    CPackedWriter* packed_writer) {
  try {
    auto truePath = std::string(path);
    auto factory = std::make_shared<milvus_storage::FileSystemFactory>();
    auto conf = milvus_storage::StorageConfig();
    conf.uri = "file:///tmp/";
    auto trueFs = factory->BuildFileSystem(conf, &truePath).value();
    auto trueSchema = arrow::ImportSchema(schema).ValueOrDie();
    auto writer =
        std::make_unique<milvus_storage::PackedRecordBatchWriter>(buffer_size, trueSchema, trueFs, truePath, conf);

    *packed_writer = writer.release();
    return 0;
  } catch (std::exception& e) {
    return -1;
  }
}

int WriteRecordBatch(CPackedWriter c_packed_writer, struct ArrowArray* array, struct ArrowSchema* schema) {
  try {
    auto packed_writer = static_cast<milvus_storage::PackedRecordBatchWriter*>(c_packed_writer);
    auto record_batch = arrow::ImportRecordBatch(array, schema).ValueOrDie();
    auto status = packed_writer->Write(record_batch);
    if (!status.ok()) {
      return -1;
    }
    return 0;
  } catch (std::exception& e) {
    return -1;
  }
}

int Close(CPackedWriter c_packed_writer, CColumnIndexGroups c_column_index_groups) {
  try {
    if (c_packed_writer == nullptr) {
      return 0;
    }
    auto packed_writer = static_cast<milvus_storage::PackedRecordBatchWriter*>(c_packed_writer);
    auto column_index_groups = static_cast<milvus_storage::ColumnIndexGroups*>(c_column_index_groups);
    auto status = packed_writer->Close(c_column_index_groups);
    delete packed_writer;
    if (!status.ok()) {
      return -1;
    }
    return 0;
  } catch (std::exception& e) {
    return -1;
  }
}