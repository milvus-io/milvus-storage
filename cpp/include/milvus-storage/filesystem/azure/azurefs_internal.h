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

#include <optional>
#include <string_view>

#include "arrow/result.h"

#include "milvus-storage/common/extend_status.h"

namespace Azure::Storage::Files::DataLake {
class DataLakeFileSystemClient;
class DataLakeServiceClient;
}  // namespace Azure::Storage::Files::DataLake

namespace milvus_storage::fs {

using arrow::Result;

struct AzureOptions;

namespace internal {

/// \brief Map an Azure failure onto the shared ExtendStatusCode taxonomy.
///
/// Producer owns classification: the Azure filesystem is the only layer that
/// still has the typed `RequestFailedException`, so the transient-vs-permanent
/// verdict has to be made here. Anything this returns `nullopt` for stays an
/// untagged `Status::IOError` and reaches segcore as StorageError/2044 --
/// permanent and non-retriable, the same conservative default the S3 path uses.
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
std::optional<milvus_storage::ExtendStatusCode> ClassifyAzureError(int http_status,
                                                                   std::string_view error_code,
                                                                   bool transport_failure);

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
