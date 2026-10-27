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
    const char* federated_token = std::getenv("AZURE_FEDERATED_TOKEN_FILE");
    const char* client_secret = std::getenv("AZURE_CLIENT_SECRET");
    if (federated_token != nullptr && strlen(federated_token) > 0) {
      // Workload Identity — checked first to preserve existing WI deployments that may
      // also have AZURE_CLIENT_SECRET set.
      LOG_STORAGE_INFO_ << "AzureFileSystemProducer: credential=WorkloadIdentity";
      const char* azure_client_id = std::getenv("AZURE_CLIENT_ID");
      if (azure_client_id == nullptr || strlen(azure_client_id) == 0) {
        return arrow::Status::Invalid("AZURE_CLIENT_ID environment variable is not set");
      }
      const char* azure_tenant_id = std::getenv("AZURE_TENANT_ID");
      if (azure_tenant_id == nullptr || strlen(azure_tenant_id) == 0) {
        return arrow::Status::Invalid("AZURE_TENANT_ID environment variable is not set");
      }
      ARROW_RETURN_NOT_OK(options.ConfigureWorkloadIdentityCredential());
    } else if (client_secret != nullptr && strlen(client_secret) > 0) {
      // Service Principal (client secret) — EnvironmentCredential reads
      // AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID.
      const char* azure_client_id = std::getenv("AZURE_CLIENT_ID");
      if (azure_client_id == nullptr || strlen(azure_client_id) == 0) {
        return arrow::Status::Invalid(
            "AZURE_CLIENT_ID environment variable is not set (required when AZURE_CLIENT_SECRET is set)");
      }
      const char* azure_tenant_id = std::getenv("AZURE_TENANT_ID");
      if (azure_tenant_id == nullptr || strlen(azure_tenant_id) == 0) {
        return arrow::Status::Invalid(
            "AZURE_TENANT_ID environment variable is not set (required when AZURE_CLIENT_SECRET is set)");
      }
      LOG_STORAGE_INFO_ << "AzureFileSystemProducer: credential=EnvironmentCredential(SPN)"
                        << " tenant_id=set client_id=set client_secret=set";
      ARROW_RETURN_NOT_OK(options.ConfigureEnvironmentCredential());
    } else {
      // Managed Identity
      LOG_STORAGE_INFO_ << "AzureFileSystemProducer: credential=ManagedIdentity";
      const char* client_id = std::getenv("AZURE_CLIENT_ID");
      if (client_id == nullptr || strlen(client_id) == 0) {
        return arrow::Status::Invalid("AZURE_CLIENT_ID environment variable is not set");
      }
      ARROW_RETURN_NOT_OK(options.ConfigureManagedIdentityCredential(std::string(client_id)));
    }
  } else {
    // need azure secret key without iam
    assert(!config_.access_key_value.empty());
    ARROW_RETURN_NOT_OK(options.ConfigureAccountKeyCredential(config_.access_key_value));
  }

  return milvus_storage::fs::AzureFileSystem::Make(options);
}

}  // namespace milvus_storage
