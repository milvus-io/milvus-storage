// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "arrow/result.h"

#include "milvus-storage/common/extend_status.h"

namespace Azure::Storage::Files::DataLake {
class DataLakeFileSystemClient;
class DataLakeServiceClient;
}  // namespace Azure::Storage::Files::DataLake

namespace Azure::Core {
class RequestFailedException;
}  // namespace Azure::Core

namespace milvus_storage::fs {

using arrow::Result;

struct AzureOptions;

namespace internal {

enum class AzureResourceKind : uint8_t {
  Unknown,
  Container,
};

/// \brief Map an Azure failure onto the shared ExtendStatusCode taxonomy.
///
/// Producer owns classification: the Azure filesystem is the only layer that
/// still has the typed `RequestFailedException`, so the retryable-vs-system
/// verdict has to be made here. Anything this returns `nullopt` for stays an
/// untagged `Status::IOError` and reaches segcore as the conservative,
/// non-retryable StorageError/2044 fallback used by the S3 path too.
///
/// Takes the raw HTTP status rather than the SDK enum so it can be unit-tested
/// without an Azure account or a synthesized SDK exception.
///
/// \param http_status the HTTP status code, or 0 when the exception carries no
///        response. 0 alone does NOT mean "network failure": Azure's
///        `RequestFailedException(std::string)` also leaves `StatusCode` at
///        `None`, and `PollUntilDone` raises exactly that when a copy operation
///        ends in a failed state. Only `transport_failure` may be treated as
///        retriable.
/// \param error_code the Azure `ErrorCode` string, used only where the HTTP
///        status alone is ambiguous (409 and 503).
/// \param transport_failure true only for `Azure::Core::Http::TransportException`
///        -- the request never reached the service (connection refused/reset,
///        DNS, TLS). `http_status` is ignored when this is set.
/// \param reason_phrase the exception's ReasonPhrase, used only to tell a
///        missing CONTAINER from a missing blob on a 404. Azure sometimes
///        leaves ErrorCode empty and says it there instead -- the same reason
///        IsContainerNotFound checks both. Defaulted so the many call sites
///        that only have a status and an error code keep compiling; omitting it
///        costs nothing but the container/blob distinction.
/// \param resource_kind identifies container-level operations when a proxy has
///        removed both ErrorCode and ReasonPhrase from a 404 response.
std::optional<milvus_storage::ExtendStatusCode> ClassifyAzureError(
    int http_status,
    std::string_view error_code,
    bool transport_failure,
    std::string_view reason_phrase = {},
    AzureResourceKind resource_kind = AzureResourceKind::Unknown);

/// TODO(stringly-typed): the LoonBroker* error codes are stringly-typed. A
/// broker failure travels through Azure's HTTP-only policy interface as a magic
/// string in `x-ms-error-code`, then is matched back by string comparison in
/// BrokerErrorCodeToExtendStatus. A typo or a drift between the producer
/// (azure_sas_token_policy.cpp) and the consumer silently degrades the
/// classification to "unclassified / non-retriable" with no compile-time
/// protection. Replace with a typed code once the SDK pipeline exposes a
/// channel that carries an ExtendStatusCode end to end.
///
/// Every synthetic response the SAS policy stamps when it cannot obtain a token
/// carries an error code under this prefix. The prefix is what lets
/// ClassifyAzureError short-circuit: a LoonBroker* code means "this failure
/// came from the local credential broker, not from Azure Storage", so it is
/// decoded by BrokerErrorCodeToExtendStatus instead of the HTTP-status switch
/// that handles real Azure responses.
inline constexpr const char* kSyntheticBrokerErrorCodePrefix = "LoonBroker";

/// The broker rejected our credentials (401/403). A genuine verdict, not a
/// fallback: only a broker that actually rejected us reaches this code.
inline constexpr const char* kSyntheticBrokerAccessDeniedErrorCode = "LoonBrokerAccessDenied";

/// The broker request timed out or its context deadline fired.
inline constexpr const char* kSyntheticBrokerTimeoutErrorCode = "LoonBrokerTimeout";

/// The broker throttled us.
inline constexpr const char* kSyntheticBrokerThrottlingErrorCode = "LoonBrokerThrottling";

/// The broker answered 5xx or is otherwise transiently unwell.
inline constexpr const char* kSyntheticBrokerServiceUnavailableErrorCode = "LoonBrokerServiceUnavailable";

/// The broker could not be reached at all (connection refused/reset, DNS, TLS).
inline constexpr const char* kSyntheticBrokerNetworkErrorCode = "LoonBrokerNetworkFailure";

/// Carries an unclassified broker-side failure through Azure's HTTP-only
/// policy interface without relabelling it as authentication or service
/// availability. Deliberately System, not retryable: the only producer left is
/// `catch (...)`, i.e. an exception we did not anticipate, which is almost
/// always our bug -- replaying a bug only reproduces it, and the taxonomy never
/// invents retriability.
inline constexpr const char* kSyntheticBrokerUnexpectedErrorCode = "LoonBrokerUnexpectedFailure";

/// Carries an allocation failure through Azure's HTTP-only policy interface.
/// AzureExceptionToStatus restores it to arrow::Status::OutOfMemory so the
/// direct C++ boundary can map it to MemAllocateFailed.
inline constexpr const char* kSyntheticBrokerOutOfMemoryErrorCode = "LoonBrokerOutOfMemory";

/// Map a LoonBroker* error code back to the ExtendStatusCode the SAS policy
/// started from. Returns nullopt for the OOM/unexpected codes, which are not
/// ExtendStatusCode conditions -- AzureExceptionToStatus handles them before
/// the general classifier is reached.
std::optional<milvus_storage::ExtendStatusCode> BrokerErrorCodeToExtendStatus(std::string_view error_code);

/// Translate the typed Azure exception while its ErrorCode and dynamic type
/// are still available. Public only for unit tests; production call sites use
/// the variadic ExceptionToStatus wrapper in azurefs.cc.
arrow::Status AzureExceptionToStatus(const Azure::Core::RequestFailedException& exception,
                                     std::string prefix,
                                     AzureResourceKind resource_kind = AzureResourceKind::Unknown);

enum class HierarchicalNamespaceSupport {
  kUnknown = 0,
  kContainerNotFound = 1,
  kDisabled = 2,
  kEnabled = 3,
};

/// \brief Performs a request to check if the storage account has Hierarchical
/// Namespace support enabled.
///
/// This check requires a DataLakeFileSystemClient for any container of the
/// storage account. If the container doesn't exist yet, we just forward that
/// error to the caller (kContainerNotFound) since that's a proper error to the operation
/// on that container anyways -- no need to try again with or without the knowledge of
/// Hierarchical Namespace support.
///
/// Hierarchical Namespace support can't easily be changed after the storage account is
/// created and the feature is shared by all containers in the storage account.
/// This means the result of this check can (and should!) be cached as soon as
/// it returns a successful result on any container of the storage account (see
/// AzureFileSystem::Impl).
///
/// The check consists of a call to DataLakeFileSystemClient::GetAccessControlList()
/// on the root directory of the container. An approach taken by the Hadoop Azure
/// project [1]. A more obvious approach would be to call
/// BlobServiceClient::GetAccountInfo(), but that endpoint requires elevated
/// permissions [2] that we can't generally rely on.
///
/// [1]:
/// https://github.com/apache/hadoop/blob/7c6af6a5f626d18d68b656d085cc23e4c1f7a1ef/hadoop-tools/hadoop-azure/src/main/java/org/apache/hadoop/fs/azurebfs/AzureBlobFileSystemStore.java#L356.
/// [2]:
/// https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob-service-properties?tabs=azure-ad#authorization
///
/// IMPORTANT: If the result is kEnabled or kDisabled, it doesn't necessarily mean that
/// the container exists.
///
/// \param adlfs_client A DataLakeFileSystemClient for a container of the storage
/// account.
/// \return kEnabled/kDisabled/kContainerNotFound (kUnknown is never
/// returned).
ARROW_EXPORT Result<HierarchicalNamespaceSupport> CheckIfHierarchicalNamespaceIsEnabled(
    const Azure::Storage::Files::DataLake::DataLakeFileSystemClient& adlfs_client,
    const milvus_storage::fs::AzureOptions& options);

}  // namespace internal
}  // namespace milvus_storage::fs
