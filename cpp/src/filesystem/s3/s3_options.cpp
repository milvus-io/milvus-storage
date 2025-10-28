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

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <arrow/util/logging.h>
#include <arrow/util/uri.h>

namespace milvus_storage {

MilvusS3GlobalOptions MilvusS3GlobalOptions::Defaults() {
  auto log_level = arrow::fs::S3LogLevel::Fatal;
  int num_event_loop_threads = 1;
  // Extract, trim, and downcase the value of the environment variable
  auto value = GetEnvVar("ARROW_S3_LOG_LEVEL")
                   .Map(arrow::internal::AsciiToLower)
                   .Map(arrow::internal::TrimString)
                   .ValueOr("fatal");
  if (value == "fatal") {
    log_level = arrow::fs::S3LogLevel::Fatal;
  } else if (value == "error") {
    log_level = arrow::fs::S3LogLevel::Error;
  } else if (value == "warn") {
    log_level = arrow::fs::S3LogLevel::Warn;
  } else if (value == "info") {
    log_level = arrow::fs::S3LogLevel::Info;
  } else if (value == "debug") {
    log_level = arrow::fs::S3LogLevel::Debug;
  } else if (value == "trace") {
    log_level = arrow::fs::S3LogLevel::Trace;
  } else if (value == "off") {
    log_level = arrow::fs::S3LogLevel::Off;
  }

  return MilvusS3GlobalOptions{log_level, 1};
}

}  // namespace milvus_storage
