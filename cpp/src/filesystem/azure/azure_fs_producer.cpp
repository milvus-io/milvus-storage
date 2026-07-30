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

#include <cstdlib>
#include <cassert>

#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/filesystem/azure/azure_sas_token_policy.h"
#include "milvus-storage/common/log.h"

#include "milvus-storage/common/macro.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/azure/azure_fs_producer.h"

namespace milvus_storage {

arrow::Result<ArrowFileSystemPtr> AzureFileSystemProducer::Make() {
  if (!config_.tls_min_version.empty()) {
    LOG_STORAGE_WARNING_ << "tls_min_version is not yet supported for Azure filesystem. "
                         << "Requested version: " << config_.tls_min_version << ". Ignoring.";
  }

  milvus_storage::fs::AzureOptions options;
  assert(!config_.access_key_id.empty());
  options.account_name = config_.access_key_id;

  if (!config_.address.empty()) {
    const char* azurite_env = std::getenv("USE_AZURITE");
    // use the azurite
    if (azurite_env && (std::string(azurite_env) == "true")) {
      options.blob_storage_authority = config_.address;
      options.dfs_storage_authority = config_.address;
    } else {  // use the azure cloud
      options.blob_storage_authority = ".blob." + config_.address;
      options.dfs_storage_authority = ".dfs." + config_.address;
    }
  }
  if (!config_.use_ssl) {
    options.blob_storage_scheme = "http";
    options.dfs_storage_scheme = "http";
  }
  options.background_writes = config_.background_writes;

  if (config_.IsAzureCredentialBrokerEnabled()) {
    fs::AzureSasBrokerConfig broker_config{
        .endpoint = config_.azure_credential_endpoint,
        .region = config_.region,
        .bucket = config_.bucket_name,
        .client_id = config_.azure_client_id,
        .tenant_id = config_.azure_tenant_id,
        .account_name = config_.access_key_id,
        .duration_seconds = config_.load_frequency,
        .request_timeout_ms = config_.request_timeout_ms,
    };
    ARROW_RETURN_NOT_OK(
        options.ConfigureSASTokenPolicy(std::make_shared<fs::AzureSasTokenPolicy>(std::move(broker_config))));
  } else if (config_.use_iam) {
    const char* federated_token = getenv("AZURE_FEDERATED_TOKEN_FILE");
    if (federated_token != nullptr && strlen(federated_token) > 0) {
      // Workload Identity
      if (std::getenv("AZURE_CLIENT_ID") == nullptr) {
        return arrow::Status::Invalid("AZURE_CLIENT_ID environment variable is not set");
      }
      if (std::getenv("AZURE_TENANT_ID") == nullptr) {
        return arrow::Status::Invalid("AZURE_TENANT_ID environment variable is not set");
      }
      ARROW_RETURN_NOT_OK(options.ConfigureWorkloadIdentityCredential());
    } else {
      // Managed Identity
      const char* client_id = std::getenv("AZURE_CLIENT_ID");
      if (client_id == nullptr) {
        return arrow::Status::Invalid("AZURE_CLIENT_ID environment variable is not set");
      }
      ARROW_RETURN_NOT_OK(options.ConfigureManagedIdentityCredential(std::string(client_id)));
    }
  } else {
    // need azure secret key without iam
    assert(!config_.access_key_value.empty());
    ARROW_RETURN_NOT_OK(options.ConfigureAccountKeyCredential(config_.access_key_value));
  }

  ARROW_ASSIGN_OR_RAISE(auto fs, milvus_storage::fs::AzureFileSystem::Make(options));
  return std::make_shared<FileSystemProxy>(config_.bucket_name, fs);
}

}  // namespace milvus_storage
