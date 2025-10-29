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

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/sts/STSClient.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/result.h>
#include <arrow/util/uri.h>
#include "milvus-storage/filesystem/s3/s3_internal.h"

namespace milvus_storage {

struct ARROW_EXPORT MilvusS3GlobalOptions {
  /// The log level for S3-originating messages.
  arrow::fs::S3LogLevel log_level;

  /// The number of threads to configure when creating AWS' I/O event loop
  ///
  /// Defaults to 1 as recommended by AWS' doc when the # of connections is
  /// expected to be, at most, in the hundreds
  ///
  /// For more details see Aws::Crt::Io::EventLoopGroup
  int num_event_loop_threads = 1;

  /// Whether to install a process-wide SIGPIPE handler
  ///
  /// The AWS SDK may sometimes emit SIGPIPE signals for certain errors;
  /// by default, they would abort the current process.
  /// This option, if enabled, will install a process-wide signal handler
  /// that logs and otherwise ignore incoming SIGPIPE signals.
  ///
  /// This option has no effect on Windows.
  bool install_sigpipe_handler = false;

  /// \brief AWS SDK wide options for http
  Aws::HttpOptions http_options;

  /// \brief Override default http options
  bool override_default_http_options = false;

  /// \brief Initialize with default options
  ///
  /// For log_level, this method first tries to extract a suitable value from the
  /// environment variable ARROW_S3_LOG_LEVEL.
  static MilvusS3GlobalOptions Defaults();
};

}  // namespace milvus_storage
