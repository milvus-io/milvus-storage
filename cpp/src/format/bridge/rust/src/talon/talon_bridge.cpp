// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "talon/talon_bridge.h"

#include <exception>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <stdexcept>
#include <utility>
#include <type_traits>

#include <arrow/status.h>
#include <arrow/util/io_util.h>
#include <nlohmann/json.hpp>

#include "runtime/bridge_util.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/credential_broker.h"

namespace milvus_storage::talon {
namespace {

using TalonIoCallback = void (*)(void* context, int32_t error_code, uint64_t value, const char* error_msg);

// Callback records borrow Rust-owned storage for the duration of the call.
struct TalonRawRequest {
  const uint8_t* uri;
  size_t uri_len;
  uint64_t size;
  const uint8_t* version;
  size_t version_len;
};
struct TalonRawLoadResult {
  uint64_t size;
  uint64_t blocks;
};
struct TalonRawFailure {
  uint64_t index;
  bool uncertain;
  const char* error;
};
struct TalonRawResult {
  int32_t code;
  const char* error;
  uint64_t size;
  const char* version;
  uint64_t blocks;
  const TalonRawLoadResult* results;
  size_t results_len;
  const TalonRawFailure* failures;
  size_t failures_len;
};
using TalonRawCallback = void (*)(void*, const TalonRawResult*);

extern "C" {

void talon_object_stat_async(const void* reader, TalonIoCallback callback, void* context);

void talon_object_read_async(
    const void* reader, uint64_t offset, uint64_t length, uint8_t* dst, TalonIoCallback callback, void* context);

void talon_raw_stat_async(
    const void* client, const uint8_t* uri, size_t uri_len, TalonRawCallback callback, void* context);
void talon_raw_read_async(const void* client,
                          const uint8_t* uri,
                          size_t uri_len,
                          uint64_t offset,
                          uint64_t length,
                          uint8_t* dst,
                          bool has_stat,
                          uint64_t size,
                          const uint8_t* version,
                          size_t version_len,
                          TalonIoCallback callback,
                          void* context);
void talon_raw_load_async(const void* client,
                          const TalonRawRequest* requests,
                          size_t count,
                          bool batch,
                          TalonRawCallback callback,
                          void* context);

void talon_free_error_string(char* ptr);

}  // extern "C"

struct TalonIoCompletion {
  TalonIoCompletion(arrow::Future<int64_t> future, const char* operation)
      : future(std::move(future)), operation(operation) {}

  arrow::Future<int64_t> future;
  const char* const operation;
};

}  // namespace

namespace internal {

arrow::Result<int64_t> ToArrowIoResult(const char* operation,
                                       int32_t error_code,
                                       uint64_t value,
                                       const char* error_msg) {
  if (error_code == 0) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return arrow::Status::IOError(operation, ": Talon returned a value larger than INT64_MAX");
    }
    return static_cast<int64_t>(value);
  }

  const std::string message = error_msg == nullptr ? "unknown Talon error" : error_msg;
  auto status = arrow::Status::IOError(operation, " (Talon error code ", error_code, "): ", message);
  switch (static_cast<ffi::TalonErrorCode>(error_code)) {
    case ffi::TalonErrorCode::InvalidArgument:
      return arrow::Status::Invalid(status.message());
    case ffi::TalonErrorCode::NotFound:
      return status.WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
    case ffi::TalonErrorCode::Timeout:
      return MakeExtendError(ExtendStatusCode::StorageTransientTimeout, status.message(), message);
    case ffi::TalonErrorCode::Unavailable:
      return MakeExtendError(ExtendStatusCode::StorageTransientService, status.message(), message);
    case ffi::TalonErrorCode::RateLimited:
      return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, status.message(), message);
    case ffi::TalonErrorCode::VersionMismatch:
      return MakeExtendError(ExtendStatusCode::AwsErrorPreConditionFailed, status.message(), message);
    case ffi::TalonErrorCode::NonRetryable:
      return MakeExtendError(ExtendStatusCode::AwsErrorNonRetryable, status.message(), message);
    default:
      return status;
  }
}

}  // namespace internal

namespace {

template <typename T>
void TalonRawCallbackImpl(void* context, const TalonRawResult* result) noexcept {
  std::unique_ptr<arrow::Future<T>> future(static_cast<arrow::Future<T>*>(context));
  // MarkFinished may invoke consumers after publishing the result. Never retry
  // completion; noexcept terminates if it throws instead of crossing into Rust.
  future->MarkFinished([&]() -> arrow::Result<T> {
    try {
      const auto status =
          result->code == 0 ? arrow::Status::OK()
                            : internal::ToArrowIoResult("Talon raw operation", result->code, 0, result->error).status();
      if (result->code < 0) {
        return status;
      }
      if constexpr (std::is_same_v<T, TalonBatchLoadResult>) {
        TalonBatchLoadResult value{status, {}, {}};
        value.results.reserve(result->results_len);
        for (size_t i = 0; i < result->results_len; ++i) {
          value.results.push_back({result->results[i].size, result->results[i].blocks});
        }
        value.failed_files.reserve(result->failures_len);
        for (size_t i = 0; i < result->failures_len; ++i) {
          const auto& failure = result->failures[i];
          value.failed_files.push_back({failure.index, failure.uncertain, failure.error});
        }
        return value;
      } else if (!status.ok()) {
        return status;
      } else if constexpr (std::is_same_v<T, TalonObjectStat>) {
        return TalonObjectStat{result->size, result->version};
      } else {
        return TalonLoadResult{result->size, result->blocks};
      }
    } catch (const std::bad_alloc& error) {
      return arrow::Status::OutOfMemory("Talon callback: ", error.what());
    } catch (const std::exception& error) {
      return arrow::Status::IOError("Talon callback: ", error.what());
    } catch (...) {
      return arrow::Status::IOError("Talon callback: unknown exception");
    }
  }());
}

void TalonIoCallbackImpl(void* context, int32_t error_code, uint64_t value, const char* error_msg) noexcept {
  // Rust invokes the callback exactly once, so this callback reclaims both the
  // completion context and any owned Rust error string.
  std::unique_ptr<TalonIoCompletion> completion(static_cast<TalonIoCompletion*>(context));
  std::unique_ptr<char, decltype(&talon_free_error_string)> error(const_cast<char*>(error_msg),
                                                                  &talon_free_error_string);
  // Keep completion outside the conversion handlers for the same one-shot guarantee.
  completion->future.MarkFinished([&]() -> arrow::Result<int64_t> {
    try {
      return internal::ToArrowIoResult(completion->operation, error_code, value, error.get());
    } catch (const std::bad_alloc& exception) {
      return arrow::Status::OutOfMemory(completion->operation,
                                        ": failed to convert callback result: ", exception.what());
    } catch (const std::exception& exception) {
      return arrow::Status::IOError(completion->operation, ": failed to convert callback result: ", exception.what());
    } catch (...) {
      return arrow::Status::IOError(completion->operation, ": failed to convert callback result: unknown exception");
    }
  }());
}

}  // namespace

arrow::Result<std::string> TalonClient::BuildCredentialBrokerIAMJson(const ArrowFileSystemConfig& config) {
  // Broker IAM ignores target identity fields, even when they are present in the filesystem config.
  nlohmann::json request = {{credential_broker::kCredentialType, credential_broker::kIam}};
  if (config.request_timeout_ms <= 0 || config.bucket_name.empty()) {
    return arrow::Status::Invalid("Talon credential Broker requires fs.bucket_name and positive fs.request_timeout_ms");
  }
  request[credential_broker::kCsp] = config.cloud_provider;
  request[credential_broker::kRegion] = config.region;
  request[credential_broker::kBucket] = config.bucket_name;
  request[credential_broker::kDurationSeconds] = config.load_frequency;
  std::string endpoint = config.address;
  bool use_virtual_host = config.use_virtual_host;
  if (config.cloud_provider == kCloudProviderAzure) {
    if (config.access_key_id.empty()) {
      return arrow::Status::Invalid("Talon Azure Broker requires fs.access_key_id");
    }
    request[credential_broker::kAzureAccountName] = config.access_key_id;
    // Match AzureFileSystemProducer: address is a cloud suffix except under Azurite.
    const char* azurite = std::getenv("USE_AZURITE");
    use_virtual_host = endpoint.empty() || azurite == nullptr || std::string(azurite) != "true";
    if (use_virtual_host) {
      endpoint = config.access_key_id + ".blob." + (endpoint.empty() ? "core.windows.net" : endpoint);
    }
  } else if (config.cloud_provider == kCloudProviderAWS) {
    if (config.region.empty() || config.load_frequency < 900 || config.load_frequency > 3600) {
      return arrow::Status::Invalid("Talon S3 Broker requires fs.region and fs.load_frequency in [900, 3600]");
    }
  } else if (config.cloud_provider != kCloudProviderGCP) {
    return arrow::Status::Invalid("Talon credential Broker supports AWS, GCP, and Azure target identities");
  }
  if (endpoint.empty()) {
    return arrow::Status::Invalid("Talon credential Broker requires fs.address");
  }
  nlohmann::json origin = {{credential_broker::kEndpoint, StorageUri::BuildEndpointUrl(endpoint, config.use_ssl)},
                           {credential_broker::kUseVirtualHost, use_virtual_host},
                           {credential_broker::kRequestTimeoutMs, config.request_timeout_ms}};
  if (!config.region.empty()) {
    origin[credential_broker::kRegion] = config.region;
  }
  // Do not send local CA file paths or reserved transport settings as effective options.
  try {
    const nlohmann::json descriptor = {{credential_broker::kUrl, config.credential_endpoint},
                                       {credential_broker::kRequest, std::move(request)},
                                       {credential_broker::kOrigin, std::move(origin)}};
    return descriptor.dump();
  } catch (const nlohmann::json::type_error& error) {
    // Preserve the serialization error code without exposing credential field contents.
    return arrow::Status::Invalid("Talon credential Broker fields must be valid UTF-8 (JSON error ", error.id, ")");
  }
}

arrow::Result<std::string> TalonClient::BuildCredentialBrokerARNJson(const ArrowFileSystemConfig& config) {
  nlohmann::json request = nlohmann::json::object();
  if (config.request_timeout_ms <= 0 || config.bucket_name.empty()) {
    return arrow::Status::Invalid("Talon credential Broker requires fs.bucket_name and positive fs.request_timeout_ms");
  }
  request[credential_broker::kCsp] = config.cloud_provider;
  request[credential_broker::kRegion] = config.region;
  request[credential_broker::kBucket] = config.bucket_name;
  request[credential_broker::kDurationSeconds] = config.load_frequency;
  std::string endpoint = config.address;
  bool use_virtual_host = config.use_virtual_host;
  if (config.cloud_provider == kCloudProviderAzure) {
    if (config.access_key_id.empty()) {
      return arrow::Status::Invalid("Talon Azure Broker requires fs.access_key_id");
    }
    if (config.azure_client_id.empty() || config.azure_tenant_id.empty() || config.region.empty()) {
      return arrow::Status::Invalid(
          "Talon Azure Broker stsAssumeRole requires fs.azure_client_id, "
          "fs.azure_tenant_id, and fs.region");
    }
    request[credential_broker::kAzureAccountName] = config.access_key_id;
    request[credential_broker::kAzureClientId] = config.azure_client_id;
    request[credential_broker::kAzureTenantId] = config.azure_tenant_id;
    // Match AzureFileSystemProducer: address is a cloud suffix except under Azurite.
    const char* azurite = std::getenv("USE_AZURITE");
    use_virtual_host = endpoint.empty() || azurite == nullptr || std::string(azurite) != "true";
    if (use_virtual_host) {
      endpoint = config.access_key_id + ".blob." + (endpoint.empty() ? "core.windows.net" : endpoint);
    }
  } else if (config.cloud_provider == kCloudProviderAWS) {
    if (config.region.empty() || config.load_frequency < 900 || config.load_frequency > 3600) {
      return arrow::Status::Invalid("Talon S3 Broker requires fs.region and fs.load_frequency in [900, 3600]");
    }
    if (config.role_arn.empty()) {
      return arrow::Status::Invalid("Talon S3 Broker stsAssumeRole requires fs.role_arn");
    }
    request[credential_broker::kRoleArn] = config.role_arn;
    if (!config.external_id.empty()) {
      request[credential_broker::kExternalId] = config.external_id;
    }
  } else if (config.cloud_provider == kCloudProviderGCP) {
    if (config.gcp_target_service_account.empty()) {
      return arrow::Status::Invalid("Talon GCP Broker stsAssumeRole requires fs.gcp_target_service_account");
    }
    request[credential_broker::kGcpServiceAccountEmail] = config.gcp_target_service_account;
  } else {
    return arrow::Status::Invalid("Talon credential Broker supports AWS, GCP, and Azure target identities");
  }
  if (endpoint.empty()) {
    return arrow::Status::Invalid("Talon credential Broker requires fs.address");
  }
  nlohmann::json origin = {{credential_broker::kEndpoint, StorageUri::BuildEndpointUrl(endpoint, config.use_ssl)},
                           {credential_broker::kUseVirtualHost, use_virtual_host},
                           {credential_broker::kRequestTimeoutMs, config.request_timeout_ms}};
  if (!config.region.empty()) {
    origin[credential_broker::kRegion] = config.region;
  }
  // Do not send local CA file paths or reserved transport settings as effective options.
  try {
    const nlohmann::json descriptor = {{credential_broker::kUrl, config.credential_endpoint},
                                       {credential_broker::kRequest, std::move(request)},
                                       {credential_broker::kOrigin, std::move(origin)}};
    return descriptor.dump();
  } catch (const nlohmann::json::type_error& error) {
    // Preserve the serialization error code without exposing credential field contents.
    return arrow::Status::Invalid("Talon credential Broker fields must be valid UTF-8 (JSON error ", error.id, ")");
  }
}

arrow::Result<std::shared_ptr<TalonClient>> TalonClient::Make(const ArrowFileSystemConfig& config) {
  std::string credential_broker_json;
  if (!config.credential_endpoint.empty()) {
    if (config.use_iam) {
      ARROW_ASSIGN_OR_RAISE(credential_broker_json, BuildCredentialBrokerIAMJson(config));
    } else {
      ARROW_ASSIGN_OR_RAISE(credential_broker_json, BuildCredentialBrokerARNJson(config));
    }
  }
  try {
    return CatchRustResult<std::shared_ptr<TalonClient>>("Failed to create Talon client", [&]() {
      auto impl = ffi::new_talon_client(config.talon_coordinator, config.talon_block_size,
                                        config.talon_max_idle_per_addr, config.talon_namespace, credential_broker_json);
      return std::shared_ptr<TalonClient>(new TalonClient(std::move(impl)));
    });
  } catch (const std::invalid_argument& error) {
    // CXX rejects non-UTF-8 strings before entering Rust; its diagnostic contains no input bytes.
    return arrow::Status::Invalid("Invalid Talon client options: ", error.what());
  }
}

arrow::Result<std::shared_ptr<TalonClient>> TalonClient::WithNamespace(std::string_view namespace_name) const {
  try {
    auto impl =
        impl_->with_namespace(rust::Str(namespace_name.empty() ? "" : namespace_name.data(), namespace_name.size()));
    return std::shared_ptr<TalonClient>(new TalonClient(std::move(impl)));
  } catch (const std::invalid_argument& error) {
    return arrow::Status::Invalid("Invalid Talon request options: ", error.what());
  }
}

arrow::Result<std::shared_ptr<TalonClient>> TalonClient::WithCredentialBroker(
    std::string_view credential_broker) const {
  try {
    auto impl = impl_->with_credential_broker(
        rust::Str(credential_broker.empty() ? "" : credential_broker.data(), credential_broker.size()));
    return std::shared_ptr<TalonClient>(new TalonClient(std::move(impl)));
  } catch (const std::invalid_argument& error) {
    return arrow::Status::Invalid("Invalid Talon request options: ", error.what());
  }
}

arrow::Result<TalonObjectReader> TalonClient::OpenObject(const std::string& cloud_provider,
                                                         const std::string& bucket,
                                                         const std::string& key,
                                                         const std::optional<TalonObjectStat>& initial_stat) const {
  return CatchRustResult<TalonObjectReader>("Failed to open Talon object", [&]() {
    const bool has_initial_stat = initial_stat.has_value();
    const std::string empty_version;
    return TalonObjectReader(ffi::open_talon_object(*impl_, cloud_provider, bucket, key, has_initial_stat,
                                                    has_initial_stat ? initial_stat->size : 0,
                                                    has_initial_stat ? initial_stat->version : empty_version));
  });
}

arrow::Future<int64_t> TalonClient::ReadAsync(const std::string& uri,
                                              uint64_t offset,
                                              uint64_t length,
                                              uint8_t* dst,
                                              const std::optional<TalonObjectStat>& initial_stat) const {
  auto future = arrow::Future<int64_t>::Make();
  auto completion = std::make_unique<TalonIoCompletion>(future, "Failed to read Talon URI");
  const std::string empty_version;
  const auto& version = initial_stat ? initial_stat->version : empty_version;
  talon_raw_read_async(&*impl_, reinterpret_cast<const uint8_t*>(uri.data()), uri.size(), offset, length, dst,
                       initial_stat.has_value(), initial_stat ? initial_stat->size : 0,
                       reinterpret_cast<const uint8_t*>(version.data()), version.size(), TalonIoCallbackImpl,
                       completion.release());
  return future;
}
arrow::Result<int64_t> TalonClient::Read(const std::string& uri,
                                         uint64_t offset,
                                         uint64_t length,
                                         uint8_t* dst,
                                         const std::optional<TalonObjectStat>& stat) const {
  return ReadAsync(uri, offset, length, dst, stat).result();
}
arrow::Future<TalonObjectStat> TalonClient::StatAsync(const std::string& uri) const {
  auto future = arrow::Future<TalonObjectStat>::Make();
  auto completion = std::make_unique<arrow::Future<TalonObjectStat>>(future);
  talon_raw_stat_async(&*impl_, reinterpret_cast<const uint8_t*>(uri.data()), uri.size(),
                       TalonRawCallbackImpl<TalonObjectStat>, completion.release());
  return future;
}
arrow::Result<TalonObjectStat> TalonClient::Stat(const std::string& uri) const { return StatAsync(uri).result(); }
arrow::Future<TalonLoadResult> TalonClient::LoadAsync(const TalonLoadRequest& request) const {
  auto future = arrow::Future<TalonLoadResult>::Make();
  auto completion = std::make_unique<arrow::Future<TalonLoadResult>>(future);
  const TalonRawRequest raw{reinterpret_cast<const uint8_t*>(request.uri.data()), request.uri.size(), request.size,
                            reinterpret_cast<const uint8_t*>(request.version.data()), request.version.size()};
  talon_raw_load_async(&*impl_, &raw, 1, false, TalonRawCallbackImpl<TalonLoadResult>, completion.release());
  return future;
}
arrow::Result<TalonLoadResult> TalonClient::Load(const TalonLoadRequest& request) const {
  return LoadAsync(request).result();
}
arrow::Future<TalonBatchLoadResult> TalonClient::BatchLoadAsync(const std::vector<TalonLoadRequest>& requests) const {
  std::vector<TalonRawRequest> raw;
  raw.reserve(requests.size());
  for (const auto& request : requests) {
    raw.push_back({reinterpret_cast<const uint8_t*>(request.uri.data()), request.uri.size(), request.size,
                   reinterpret_cast<const uint8_t*>(request.version.data()), request.version.size()});
  }
  auto future = arrow::Future<TalonBatchLoadResult>::Make();
  auto completion = std::make_unique<arrow::Future<TalonBatchLoadResult>>(future);
  talon_raw_load_async(&*impl_, raw.data(), raw.size(), true, TalonRawCallbackImpl<TalonBatchLoadResult>,
                       completion.release());
  return future;
}
arrow::Result<TalonBatchLoadResult> TalonClient::BatchLoad(const std::vector<TalonLoadRequest>& requests) const {
  return BatchLoadAsync(requests).result();
}

int64_t TalonObjectReader::KnownSize() const { return ffi::talon_object_known_size(*impl_); }

arrow::Future<int64_t> TalonObjectReader::StatAsync() const {
  auto future = arrow::Future<int64_t>::Make();
  auto completion = std::make_unique<TalonIoCompletion>(future, "Failed to stat Talon object");
  auto* const raw_completion = completion.release();
  talon_object_stat_async(static_cast<const void*>(&*impl_), TalonIoCallbackImpl, static_cast<void*>(raw_completion));
  return future;
}

arrow::Future<int64_t> TalonObjectReader::ReadAtAsync(uint64_t offset, uint64_t length, uint8_t* dst) const {
  auto future = arrow::Future<int64_t>::Make();
  auto completion = std::make_unique<TalonIoCompletion>(future, "Failed to read Talon object");
  auto* const raw_completion = completion.release();
  talon_object_read_async(static_cast<const void*>(&*impl_), offset, length, dst, TalonIoCallbackImpl,
                          static_cast<void*>(raw_completion));
  return future;
}

}  // namespace milvus_storage::talon
