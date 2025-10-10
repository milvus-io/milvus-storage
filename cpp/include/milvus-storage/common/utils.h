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

#include <arrow/filesystem/filesystem.h>
#include <arrow/type.h>
#include <arrow/result.h>
#include <memory>
#include <string>
#include <unordered_map>

#include "proto/schema_arrow.pb.h"
#include "milvus-storage/storage/options.h"

namespace milvus_storage {

arrow::Result<std::unique_ptr<schema_proto::ArrowSchema>> ToProtobufSchema(const arrow::Schema* schema);

arrow::Result<std::shared_ptr<arrow::Schema>> FromProtobufSchema(const schema_proto::ArrowSchema& schema);

std::string GetNewParquetFilePath(const std::string& path);

std::string GetManifestFilePath(const std::string& path, int64_t version);

std::string GetManifestTmpFilePath(const std::string& path, int64_t version);

arrow::Result<std::shared_ptr<arrow::Schema>> ProjectSchema(std::shared_ptr<arrow::Schema> schema,
                                                            const ReadOptions& options);

int64_t ParseVersionFromFileName(const std::string& path);

ReadOptions CreateInternalReadOptions(std::shared_ptr<arrow::Schema> schema,
                                      const SchemaOptions& schema_options,
                                      const ReadOptions& options);

std::string GetManifestDir(const std::string& path);
std::string GetScalarDataDir(const std::string& path);
std::string GetVectorDataDir(const std::string& path);
std::string GetDeleteDataDir(const std::string& path);
std::string GetBlobDir(const std::string& path);

std::string GetNewBlobFilePath(const std::string& path);

}  // namespace milvus_storage
