// Copyright 2026 Zilliz
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

// AzureFileSystemProducer::Make() configuration checks that need no Azure
// endpoint: where an external filesystem may send an attached credential.

#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <arrow/status.h>

#include "milvus-storage/filesystem/azure/azure_fs_producer.h"
#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/filesystem/fs.h"

namespace milvus_storage {

namespace {

// Isolate credential selection from the test runner's Azure environment.
class ScopedAzureEnv {
  public:
  ScopedAzureEnv() {
    for (const char* name : kNames) {
      const char* value = std::getenv(name);
      saved_[name] = value ? std::optional<std::string>(value) : std::nullopt;
      unsetenv(name);
    }
  }

  ~ScopedAzureEnv() {
    for (const auto& [name, value] : saved_) {
      if (value.has_value()) {
        setenv(name.c_str(), value->c_str(), 1);
      } else {
        unsetenv(name.c_str());
      }
    }
  }

  private:
  static constexpr const char* kNames[] = {
      "AZURE_FEDERATED_TOKEN_FILE", "AZURE_CLIENT_SECRET", "AZURE_CLIENT_ID", "AZURE_TENANT_ID",
      "AZURE_AUTHORITY_HOST",       "USE_AZURITE"};
  std::map<std::string, std::optional<std::string>> saved_;
};

constexpr const char* kAddressRule = "is not an Azure Storage authority";
constexpr const char* kSslRule = "use_ssl must be true";
// The use_iam path with no AZURE_* environment reaches the Managed Identity
// branch and fails on this check, which proves the address gate let it pass.
constexpr const char* kManagedIdentityCheck = "AZURE_CLIENT_ID environment variable is not set";

ArrowFileSystemConfig BaseConfig() {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "storageaccount";
  config.bucket_name = "container";
  config.use_ssl = true;
  return config;
}

ArrowFileSystemConfig ExternalIamConfig(const std::string& address) {
  auto config = BaseConfig();
  config.alias = "customer";
  config.use_iam = true;
  config.address = address;
  return config;
}

ArrowFileSystemConfig ExternalBrokerConfig(const std::string& address) {
  auto config = BaseConfig();
  config.alias = "customer";
  config.use_iam = false;
  config.address = address;
  config.credential_endpoint = "https://broker.internal/v1/credentials/sas";
  config.azure_client_id = "client-id";
  config.azure_tenant_id = "tenant-id";
  config.region = "eastus";
  config.request_timeout_ms = 1000;
  config.load_frequency = 900;
  return config;
}

arrow::Status Make(const ArrowFileSystemConfig& config) { return AzureFileSystemProducer(config).Make().status(); }

}  // namespace

class AzureFileSystemProducerTest : public ::testing::Test {
  protected:
  ScopedAzureEnv env_;
};

TEST_F(AzureFileSystemProducerTest, ExternalIamRejectsNonAzureAddress) {
  auto status = Make(ExternalIamConfig("storage.example.test"));
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kAddressRule), std::string::npos) << status.message();
  EXPECT_NE(status.message().find("address"), std::string::npos) << status.message();
  EXPECT_NE(status.message().find("storage.example.test"), std::string::npos) << status.message();

  // A suffix or prefix of an authority is not that authority.
  status = Make(ExternalIamConfig("evil.core.windows.net"));
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kAddressRule), std::string::npos) << status.message();
  status = Make(ExternalIamConfig("core.windows.net.example"));
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kAddressRule), std::string::npos) << status.message();
}

TEST_F(AzureFileSystemProducerTest, ExternalIamRejectsPlainHttp) {
  auto config = ExternalIamConfig("core.windows.net");
  config.use_ssl = false;
  auto status = Make(config);
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kSslRule), std::string::npos) << status.message();
}

TEST_F(AzureFileSystemProducerTest, ExternalIamAcceptsAzureStorageAuthorities) {
  for (const char* address : {"core.windows.net", "Core.Windows.NET", "core.chinacloudapi.cn", "core.usgovcloudapi.net",
                              "core.cloudapi.de", ""}) {
    auto status = Make(ExternalIamConfig(address));
    // Fails later for the missing identity environment, not for the address.
    ASSERT_TRUE(status.IsInvalid()) << address << ": " << status.ToString();
    EXPECT_NE(status.message().find(kManagedIdentityCheck), std::string::npos) << address << ": " << status.message();
    EXPECT_EQ(status.message().find(kAddressRule), std::string::npos) << address << ": " << status.message();
    EXPECT_EQ(status.message().find(kSslRule), std::string::npos) << address << ": " << status.message();
  }
}

TEST_F(AzureFileSystemProducerTest, PrimaryIamKeepsCustomAddressAndScheme) {
  auto config = BaseConfig();
  config.use_iam = true;
  config.address = "storage.example.test";
  config.use_ssl = false;
  auto status = Make(config);
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kManagedIdentityCheck), std::string::npos) << status.message();
  EXPECT_EQ(status.message().find(kAddressRule), std::string::npos) << status.message();
  EXPECT_EQ(status.message().find(kSslRule), std::string::npos) << status.message();
}

TEST_F(AzureFileSystemProducerTest, ExternalIamSelectsWorkloadIdentityWithoutAccountKey) {
  setenv("AZURE_FEDERATED_TOKEN_FILE", "/unused/projected-token", 1);
  setenv("AZURE_CLIENT_ID", "workload-client", 1);
  setenv("AZURE_TENANT_ID", "workload-tenant", 1);
  const auto config = ExternalIamConfig("");
  ASSERT_TRUE(config.access_key_value.empty());

  fs::AzureOptions expected;
  expected.account_name = config.access_key_id;
  ASSERT_TRUE(expected.ConfigureWorkloadIdentityCredential().ok());

  // Construction is lazy: no token file or Azure endpoint is needed. Compare
  // the credential kind as well as account/options, so a fallback to managed
  // identity or environment credentials cannot make this test pass.
  for (const bool with_client_secret : {false, true}) {
    SCOPED_TRACE(with_client_secret);
    if (with_client_secret) {
      setenv("AZURE_CLIENT_SECRET", "unused-client-secret", 1);
    }
    const auto result = AzureFileSystemProducer(config).Make();
    ASSERT_TRUE(result.ok()) << result.status();
    const auto azure_fs = std::dynamic_pointer_cast<fs::AzureFileSystem>(result.ValueOrDie());
    ASSERT_NE(azure_fs, nullptr);
    EXPECT_TRUE(azure_fs->options().Equals(expected));
  }
}

TEST_F(AzureFileSystemProducerTest, ExternalWorkloadIdentityRejectsMissingOrEmptyClientId) {
  setenv("AZURE_FEDERATED_TOKEN_FILE", "/unused/projected-token", 1);
  setenv("AZURE_TENANT_ID", "workload-tenant", 1);
  for (const bool empty : {false, true}) {
    SCOPED_TRACE(empty);
    if (empty) {
      setenv("AZURE_CLIENT_ID", "", 1);
    }
    const auto status = Make(ExternalIamConfig("core.windows.net"));
    ASSERT_TRUE(status.IsInvalid()) << status;
    EXPECT_NE(status.message().find("AZURE_CLIENT_ID environment variable is not set"), std::string::npos);
  }
}

TEST_F(AzureFileSystemProducerTest, ExternalWorkloadIdentityRejectsMissingOrEmptyTenantId) {
  setenv("AZURE_FEDERATED_TOKEN_FILE", "/unused/projected-token", 1);
  setenv("AZURE_CLIENT_ID", "workload-client", 1);
  for (const bool empty : {false, true}) {
    SCOPED_TRACE(empty);
    if (empty) {
      setenv("AZURE_TENANT_ID", "", 1);
    }
    const auto status = Make(ExternalIamConfig("core.windows.net"));
    ASSERT_TRUE(status.IsInvalid()) << status;
    // Managed identity only needs the client ID. Requiring the tenant here
    // proves incomplete workload identity configuration does not fall back.
    EXPECT_NE(status.message().find("AZURE_TENANT_ID environment variable is not set"), std::string::npos);
  }
}

TEST_F(AzureFileSystemProducerTest, ExternalSharedKeyAllowsCustomAddress) {
  // Shared keys sign requests; nothing reusable is attached, so the address
  // and scheme stay the user's choice.
  auto config = BaseConfig();
  config.alias = "customer";
  config.use_iam = false;
  config.access_key_value = "c2hhcmVkLWtleQ==";
  config.address = "storage.example.test";
  config.use_ssl = false;
  auto status = Make(config);
  EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST_F(AzureFileSystemProducerTest, ExternalBrokerSasFollowsTheSameRule) {
  auto status = Make(ExternalBrokerConfig("storage.example.test"));
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kAddressRule), std::string::npos) << status.message();

  auto plain_http = ExternalBrokerConfig("core.windows.net");
  plain_http.use_ssl = false;
  status = Make(plain_http);
  ASSERT_TRUE(status.IsInvalid()) << status.ToString();
  EXPECT_NE(status.message().find(kSslRule), std::string::npos) << status.message();

  // The SAS policy only contacts the Broker on the first request.
  status = Make(ExternalBrokerConfig("core.windows.net"));
  EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST_F(AzureFileSystemProducerTest, BrokerIamDoesNotRequireLocalIdentityOrTarget) {
  auto config = ExternalBrokerConfig("core.windows.net");
  config.use_iam = true;
  config.azure_client_id.clear();
  config.azure_tenant_id.clear();
  EXPECT_TRUE(Make(config).ok());

  config.credential_endpoint.clear();
  const auto status = Make(config);
  EXPECT_TRUE(status.IsInvalid()) << status;
  EXPECT_NE(status.message().find(kManagedIdentityCheck), std::string::npos);
}

}  // namespace milvus_storage
