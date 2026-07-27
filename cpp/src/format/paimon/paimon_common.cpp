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

#include "milvus-storage/format/paimon/paimon_common.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace milvus_storage::paimon {

arrow::Status ClassifyPaimonError(const std::string& context, const std::exception& error) {
  const std::string_view message{error.what()};
  // Stable marker prefixes produced by the Rust bridge for terminal errors.
  // Do not infer status from human-readable prose: wording may evolve while
  // these markers remain part of the FFI contract.
  if (message.find("[paimon:error=invalid]") != std::string_view::npos) {
    return arrow::Status::Invalid(context, ": ", error.what());
  }
  if (message.find("[paimon:error=not-implemented]") != std::string_view::npos) {
    return arrow::Status::NotImplemented(context, ": ", error.what());
  }
  return arrow::Status::IOError(context, ": ", error.what());
}

std::unordered_map<std::string, std::string> ToStorageOptions(const ArrowFileSystemConfig& config) {
  std::unordered_map<std::string, std::string> options;
  auto set = [&options](const std::string& key, const std::string& value) {
    if (!value.empty()) {
      options[key] = value;
    }
  };
  auto endpoint = [&config]() {
    return config.address.empty() ? std::string{} : StorageUri::BuildEndpointUrl(config.address, config.use_ssl);
  };

  if (config.storage_type == "local") {
    return options;
  }
  if (config.cloud_provider == kCloudProviderAWS) {
    set("s3.endpoint", endpoint());
    set("s3.region", config.region);
    options["s3.path-style-access"] = config.use_virtual_host ? "false" : "true";
    if (!config.use_iam && config.role_arn.empty()) {
      set("s3.access-key", config.access_key_id);
      set("s3.secret-key", config.access_key_value);
    }
    set("s3.assumed.role.arn", config.role_arn);
    set("s3.assumed.role.externalId", config.external_id);
    set("s3.assumed.role.session.name", config.session_name);
    return options;
  }
  if (config.cloud_provider == kCloudProviderAliyun) {
    // paimon-rust currently accepts static OSS credentials (including a
    // security token) but does not expose OSS AssumeRole fields.  Do not pass
    // role fields under invented keys: callers must provide resolved STS
    // credentials until upstream gains that provider flow.
    if (!config.role_arn.empty()) {
      throw std::runtime_error("Paimon OSS role ARN is not supported by the current paimon-rust FileIO");
    }
    set("fs.oss.endpoint", endpoint());
    set("fs.oss.accessKeyId", config.access_key_id);
    set("fs.oss.accessKeySecret", config.access_key_value);
    return options;
  }
  if (config.cloud_provider == kCloudProviderAzure) {
    set("azure.endpoint", endpoint());
    set("azure.account-name", config.access_key_id);
    if (config.use_iam) {
      if (const auto* value = std::getenv("AZURE_CLIENT_ID"))
        set("azure.client-id", value);
      if (const auto* value = std::getenv("AZURE_CLIENT_SECRET"))
        set("azure.client-secret", value);
      if (const auto* value = std::getenv("AZURE_TENANT_ID"))
        set("azure.tenant-id", value);
      if (const auto* value = std::getenv("AZURE_AUTHORITY_HOST"))
        set("azure.authority-host", value);
    } else {
      set("azure.account-key", config.access_key_value);
    }
    return options;
  }
  if (config.cloud_provider == kCloudProviderGCP) {
    if (!config.gcp_target_service_account.empty()) {
      throw std::runtime_error(
          "Paimon GCS service account impersonation is not supported by the current paimon-rust FileIO");
    }
    set("gcs.endpoint", endpoint());
    set("gcs.credential", config.gcp_credential_json);
    if (config.gcp_native_without_auth) {
      options["gcs.allow-anonymous"] = "true";
    }
    return options;
  }
  throw std::runtime_error("Unsupported cloud provider for Paimon: " + config.cloud_provider);
}

std::string ToStandardUri(const std::string& milvus_uri) {
  auto parsed = StorageUri::Parse(milvus_uri);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return milvus_uri;
  }
  auto result = StorageUri::Make(parsed.ValueOrDie(), false);
  return result.ok() ? result.ValueOrDie() : milvus_uri;
}

std::string ToMilvusUri(const std::string& standard_uri, const std::string& address) {
  if (address.empty()) {
    return standard_uri;
  }
  auto parsed = StorageUri::Parse(standard_uri, false);
  if (!parsed.ok() || parsed->scheme.empty()) {
    return standard_uri;
  }
  parsed->address = address;
  auto result = StorageUri::Make(parsed.ValueOrDie());
  return result.ok() ? result.ValueOrDie() : standard_uri;
}

}  // namespace milvus_storage::paimon
