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

#include <aws/core/Aws.h>
#include <aws/core/http/HttpTypes.h>

#include <arrow/status.h>
#include <arrow/util/logging.h>

namespace milvus_storage {

enum class S3LogLevel : int8_t { Off, Fatal, Error, Warn, Info, Debug, Trace };

struct S3GlobalOptions {
  /// The log level for S3-originating messages.
  S3LogLevel log_level;

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
  static S3GlobalOptions Defaults();
};

///
/// Global S3 initialization and finalization functions
/// These functions manage the lifecycle of the AWS SDK used by S3FileSystem.
///

/// \brief Initialize the S3 APIs with the specified set of options.
///
/// It is required to call this function at least once before using S3FileSystem.
///
/// Once this function is called you MUST call FinalizeS3 before the end of the
/// application in order to avoid a segmentation fault at shutdown.
arrow::Status InitializeS3(const S3GlobalOptions& options);

/// \brief Ensure the S3 APIs are initialized, but only if not already done.
///
/// If necessary, this will call InitializeS3() with some default options.
arrow::Status EnsureS3Initialized();

/// Whether S3 was initialized, and not finalized.
bool IsS3Initialized();

/// \brief Check if S3 is initialized and return an error if not.
///
/// This function checks if S3 has been initialized and returns an appropriate
/// error status if it has not been initialized or has been finalized.
arrow::Status CheckS3Initialized();

/// \brief Shutdown the S3 APIs.
///
/// This can wait for some S3 concurrent calls to finish so as to avoid
/// race conditions.
/// After this function has been called, all S3 calls will fail with an error.
///
/// Calls to InitializeS3() and FinalizeS3() should be serialized by the
/// application (this also applies to EnsureS3Initialized() and
/// EnsureS3Finalized()).
arrow::Status FinalizeS3();

/// Whether S3 was finalized.
bool IsS3Finalized();

/// \brief Ensure the S3 APIs are shutdown, but only if not already done.
///
/// If necessary, this will call FinalizeS3().
arrow::Status EnsureS3Finalized();

}  // namespace milvus_storage