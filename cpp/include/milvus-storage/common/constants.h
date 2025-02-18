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
#include <string>

namespace milvus_storage {

const int kReadBatchSize = 1024;

const std::string kManifestTempFileSuffix = ".manifest.tmp";
const std::string kManifestFileSuffix = ".manifest";
const std::string kManifestsDir = "versions";
const std::string kScalarDataDir = "scalar";
const std::string kVectorDataDir = "vector";
const std::string kDeleteDataDir = "delete";
const std::string kBlobDir = "blobs";
const std::string kParquetDataFileSuffix = ".parquet";
const std::string kOffsetFieldName = "__offset";

const int64_t DEFAULT_CHUNK_MANAGER_REQUEST_TIMEOUT_SEC = 10;

}  // namespace milvus_storage
