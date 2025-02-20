// Copyright 2024 Zilliz
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

#include <arrow/filesystem/s3fs.h>
#include <arrow/util/uri.h>
#include <cstdlib>
#include "milvus-storage/common/log.h"
#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"

namespace milvus_storage {

static std::unordered_map<std::string, arrow::fs::S3LogLevel> LogLevel_Map = {
    {"off", arrow::fs::S3LogLevel::Off},     {"fatal", arrow::fs::S3LogLevel::Fatal},
    {"error", arrow::fs::S3LogLevel::Error}, {"warn", arrow::fs::S3LogLevel::Warn},
    {"info", arrow::fs::S3LogLevel::Info},   {"debug", arrow::fs::S3LogLevel::Debug},
    {"trace", arrow::fs::S3LogLevel::Trace}};

class S3FileSystemProducer : public FileSystemProducer {
  public:
  S3FileSystemProducer(const ArrowFileSystemConfig& config) : config_(config) {}

  Result<ArrowFileSystemPtr> Make() override;

  arrow::fs::S3Options CreateS3Options();

  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> CreateCredentialsProvider();

  void InitializeS3();

  private:
  const ArrowFileSystemConfig& config_;
};

}  // namespace milvus_storage
