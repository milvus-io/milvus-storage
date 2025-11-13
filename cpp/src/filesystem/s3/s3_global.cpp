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

#include "milvus-storage/filesystem/s3/s3_global.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <arrow/status.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <arrow/util/string.h>
#include <arrow/util/uri.h>
#include <arrow/filesystem/path_util.h>

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
namespace milvus_storage {

S3GlobalOptions S3GlobalOptions::Defaults() {
  auto log_level = S3LogLevel::Fatal;
  int num_event_loop_threads = 1;
  // Extract, trim, and downcase the value of the environment variable
  auto value = GetEnvVar("ARROW_S3_LOG_LEVEL")
                   .Map(arrow::internal::AsciiToLower)
                   .Map(arrow::internal::TrimString)
                   .ValueOr("fatal");
  if (value == "fatal") {
    log_level = S3LogLevel::Fatal;
  } else if (value == "error") {
    log_level = S3LogLevel::Error;
  } else if (value == "warn") {
    log_level = S3LogLevel::Warn;
  } else if (value == "info") {
    log_level = S3LogLevel::Info;
  } else if (value == "debug") {
    log_level = S3LogLevel::Debug;
  } else if (value == "trace") {
    log_level = S3LogLevel::Trace;
  } else if (value == "off") {
    log_level = S3LogLevel::Off;
  }

  return S3GlobalOptions{log_level, 1};
}

// -----------------------------------------------------------------------
// AWS SDK Initialization and finalization

struct AwsInstance {
  AwsInstance() : is_initialized_(false), is_finalized_(false) {}
  ~AwsInstance() { Finalize(/*from_destructor=*/true); }

  // Returns true iff the instance was newly initialized with `options`
  arrow::Result<bool> EnsureInitialized(const S3GlobalOptions& options) {
    // NOTE: The individual accesses are atomic but the entire sequence below is not.
    // The application should serialize calls to InitializeS3() and FinalizeS3()
    // (see docstrings).
    if (is_finalized_.load()) {
      return arrow::Status::Invalid("Attempt to initialize S3 after it has been finalized");
    }
    bool newly_initialized = false;
    // EnsureInitialized() can be called concurrently by FileSystemFromUri,
    // therefore we need to serialize initialization (GH-39897).
    std::call_once(initialize_flag_, [&]() {
      bool was_initialized = is_initialized_.exchange(true);
      DCHECK(!was_initialized);
      DoInitialize(options);
      newly_initialized = true;
    });
    return newly_initialized;
  }

  bool IsInitialized() { return !is_finalized_ && is_initialized_; }

  bool IsFinalized() { return is_finalized_; }

  void Finalize(bool from_destructor = false) {
    if (is_finalized_.exchange(true)) {
      // Already finalized
      return;
    }
    auto client_finalizer = GetClientFinalizer();
    if (is_initialized_.exchange(false)) {
      // Was initialized
      if (from_destructor) {
        ARROW_LOG(WARNING) << " arrow::fs::FinalizeS3 was not called even though S3 was initialized.  "
                              "This could lead to a segmentation fault at exit";
        auto* leaked_shared_ptr = new std::shared_ptr<S3ClientFinalizer>(client_finalizer);
        ARROW_UNUSED(leaked_shared_ptr);
        return;
      }
      client_finalizer->Finalize();
#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
      EndpointProviderCache::Instance()->Reset();
#endif
      Aws::ShutdownAPI(aws_options_);
    }
  }

  private:
  void DoInitialize(const S3GlobalOptions& options) {
    Aws::Utils::Logging::LogLevel aws_log_level;

#define LOG_LEVEL_CASE(level_name)                             \
  case S3LogLevel::level_name:                                 \
    aws_log_level = Aws::Utils::Logging::LogLevel::level_name; \
    break;

    switch (options.log_level) {
      LOG_LEVEL_CASE(Fatal)
      LOG_LEVEL_CASE(Error)
      LOG_LEVEL_CASE(Warn)
      LOG_LEVEL_CASE(Info)
      LOG_LEVEL_CASE(Debug)
      LOG_LEVEL_CASE(Trace)
      default:
        aws_log_level = Aws::Utils::Logging::LogLevel::Off;
    }

#undef LOG_LEVEL_CASE

    aws_options_.loggingOptions.logLevel = aws_log_level;
    // By default the AWS SDK logs to files, log to console instead
    aws_options_.loggingOptions.logger_create_fn = [this] {
      return std::make_shared<Aws::Utils::Logging::ConsoleLogSystem>(aws_options_.loggingOptions.logLevel);
    };
    if (options.override_default_http_options) {
      aws_options_.httpOptions = options.http_options;
    }
    Aws::InitAPI(aws_options_);
  }

  Aws::SDKOptions aws_options_;
  std::atomic<bool> is_initialized_;
  std::atomic<bool> is_finalized_;
  std::once_flag initialize_flag_;
};

AwsInstance* GetAwsInstance() {
  // make sure ClientFinializer is initialized before the AwsInstance
  // so that the static object destructor is called later
  GetClientFinalizer();
  static auto instance = std::make_unique<AwsInstance>();
  return instance.get();
}

arrow::Result<bool> EnsureAwsInstanceInitialized(const S3GlobalOptions& options) {
  return GetAwsInstance()->EnsureInitialized(options);
}

arrow::Status InitializeS3(const S3GlobalOptions& options) {
  ARROW_ASSIGN_OR_RAISE(bool successfully_initialized, EnsureAwsInstanceInitialized(options));
  if (!successfully_initialized) {
    return arrow::Status::Invalid(
        "S3 was already initialized.  It is safe to use but the options passed in this "
        "call have been ignored.");
  }
  return arrow::Status::OK();
}

arrow::Status EnsureS3Initialized() { return EnsureAwsInstanceInitialized(S3GlobalOptions::Defaults()).status(); }

arrow::Status FinalizeS3() {
  // GetAwsInstance()->Finalize();
  auto instance = GetAwsInstance();
  // The AWS instance might already be destroyed in case FinalizeS3
  // is called from an atexit handler (which is a bad idea anyway as the
  // AWS SDK is not safe anymore to shutdown by this time). See GH-44071.
  if (instance == nullptr) {
    return arrow::Status::Invalid("FinalizeS3 called too late");
  }
  instance->Finalize();
  return arrow::Status::OK();
}

arrow::Status EnsureS3Finalized() { return FinalizeS3(); }

bool IsS3Initialized() { return GetAwsInstance()->IsInitialized(); }

bool IsS3Finalized() { return GetAwsInstance()->IsFinalized(); }

}  // namespace milvus_storage