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

// Azure-specific unit tests ported from Arrow v23 azurefs_test.cc.
// These test AzureOptions and AzureFileSystem initialization without
// requiring a running Azure/Azurite instance.

#include <gtest/gtest.h>
#include <arrow/result.h>
#include <azure/storage/blobs.hpp>
#include <azure/storage/files/datalake.hpp>

#include <cstdint>
#include <vector>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/filesystem/azure/azure_fs_producer.h"
#include "milvus-storage/filesystem/azure/azurefs.h"

namespace milvus_storage::fs {

namespace internal {
std::vector<arrow::Status> RunHealthyCloseAbortCloseForTest();
std::vector<arrow::Status> RunBackgroundUploadOutcomeFailureForTest(int64_t* blocks_in_progress, bool* future_finished);
}  // namespace internal

// ============================================================================
// AzureFileSystem initialization tests (no network required)
// ============================================================================

TEST(AzureFileSystem, InitializingFilesystemWithoutAccountNameFails) {
  AzureOptions options;
  ASSERT_FALSE(options.ConfigureAccountKeyCredential("account_key").ok());

  ASSERT_TRUE(options.ConfigureClientSecretCredential("tenant_id", "client_id", "client_secret").ok());
  ASSERT_FALSE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithDefaultCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureDefaultCredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithDefaultCredentialImplicitly) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  AzureOptions explicitly_default_options;
  explicitly_default_options.account_name = "dummy-account-name";
  ASSERT_TRUE(explicitly_default_options.ConfigureDefaultCredential().ok());
  ASSERT_TRUE(options.Equals(explicitly_default_options));
}

TEST(AzureFileSystem, InitializeWithAnonymousCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureAnonymousCredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithClientSecretCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureClientSecretCredential("tenant_id", "client_id", "client_secret").ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithManagedIdentityCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureManagedIdentityCredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());

  ASSERT_TRUE(options.ConfigureManagedIdentityCredential("specific-client-id").ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithCLICredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureCLICredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithWorkloadIdentityCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureWorkloadIdentityCredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InitializeWithEnvironmentCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureEnvironmentCredential().ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, OptionsCompare) {
  AzureOptions options;
  EXPECT_TRUE(options.Equals(options));
}

TEST(AzureFileSystem, BackgroundUploadChildFailureSettlesBlockAndPublishesFuture) {
  int64_t blocks_in_progress = -1;
  bool future_finished = false;
  std::vector<arrow::Status> statuses;
  ASSERT_NO_THROW(statuses = internal::RunBackgroundUploadOutcomeFailureForTest(&blocks_in_progress, &future_finished));

  EXPECT_EQ(blocks_in_progress, 0);
  // The aggregate future must always be completed by the completion that claims
  // the publish slot. Leaving it pending strands every FlushAsync/CloseAsync
  // waiter, which no later completion can rescue once the slot is consumed.
  EXPECT_TRUE(future_finished);
  ASSERT_EQ(statuses.size(), 4);
  EXPECT_TRUE(statuses[0].IsIOError()) << statuses[0].ToString();
  EXPECT_NE(statuses[0].message().find("injected Azure child upload failure"), std::string::npos)
      << statuses[0].ToString();
  EXPECT_TRUE(statuses[1].Equals(statuses[0])) << statuses[1].ToString();
  EXPECT_TRUE(statuses[2].Equals(statuses[0])) << statuses[2].ToString();
  EXPECT_TRUE(statuses[3].ok()) << statuses[3].ToString();
}

TEST(AzureFileSystem, AbortAfterSuccessfulClosePreservesIdempotentClose) {
  auto statuses = internal::RunHealthyCloseAbortCloseForTest();
  ASSERT_EQ(statuses.size(), 3);
  EXPECT_TRUE(statuses[0].ok()) << statuses[0].ToString();
  EXPECT_TRUE(statuses[1].ok()) << statuses[1].ToString();
  EXPECT_TRUE(statuses[2].ok()) << statuses[2].ToString();
}

// ============================================================================
// AzureOptions::FromUri tests (no network required)
// ============================================================================

class TestAzureOptions : public ::testing::Test {};

TEST_F(TestAzureOptions, FromUriBlobStorage) {
  AzureOptions default_options;
  std::string path;
  auto result = AzureOptions::FromUri("abfs://account.blob.core.windows.net/container/dir/blob", &path);
  ASSERT_TRUE(result.ok());
  auto options = result.ValueOrDie();
  EXPECT_EQ(options.account_name, "account");
  EXPECT_EQ(options.blob_storage_authority, default_options.blob_storage_authority);
  EXPECT_EQ(options.dfs_storage_authority, default_options.dfs_storage_authority);
  EXPECT_EQ(options.blob_storage_scheme, default_options.blob_storage_scheme);
  EXPECT_EQ(options.dfs_storage_scheme, default_options.dfs_storage_scheme);
  EXPECT_EQ(path, "container/dir/blob");
  EXPECT_EQ(options.background_writes, true);
}

TEST_F(TestAzureOptions, FromUriDfsStorage) {
  AzureOptions default_options;
  std::string path;
  auto result = AzureOptions::FromUri("abfs://file_system@account.dfs.core.windows.net/dir/file", &path);
  ASSERT_TRUE(result.ok());
  auto options = result.ValueOrDie();
  EXPECT_EQ(options.account_name, "account");
  EXPECT_EQ(options.blob_storage_authority, default_options.blob_storage_authority);
  EXPECT_EQ(options.dfs_storage_authority, default_options.dfs_storage_authority);
  EXPECT_EQ(path, "file_system/dir/file");
  EXPECT_EQ(options.background_writes, true);
}

TEST_F(TestAzureOptions, FromUriAbfs) {
  std::string path;
  auto result = AzureOptions::FromUri("abfs://account@127.0.0.1:10000/container/dir/blob", &path);
  ASSERT_TRUE(result.ok());
  auto options = result.ValueOrDie();
  EXPECT_EQ(options.account_name, "account");
  EXPECT_EQ(options.blob_storage_authority, "127.0.0.1:10000");
  EXPECT_EQ(options.dfs_storage_authority, "127.0.0.1:10000");
  EXPECT_EQ(options.blob_storage_scheme, "https");
  EXPECT_EQ(options.dfs_storage_scheme, "https");
  EXPECT_EQ(path, "container/dir/blob");
}

TEST_F(TestAzureOptions, FromUriAbfss) {
  std::string path;
  auto result = AzureOptions::FromUri("abfss://account@127.0.0.1:10000/container/dir/blob", &path);
  ASSERT_TRUE(result.ok());
  auto options = result.ValueOrDie();
  EXPECT_EQ(options.account_name, "account");
  EXPECT_EQ(options.blob_storage_authority, "127.0.0.1:10000");
  EXPECT_EQ(options.blob_storage_scheme, "https");
  EXPECT_EQ(path, "container/dir/blob");
}

TEST_F(TestAzureOptions, FromUriEnableTls) {
  std::string path;
  auto result = AzureOptions::FromUri("abfs://account@127.0.0.1:10000/container/dir/blob?enable_tls=false", &path);
  ASSERT_TRUE(result.ok());
  auto options = result.ValueOrDie();
  EXPECT_EQ(options.blob_storage_scheme, "http");
  EXPECT_EQ(options.dfs_storage_scheme, "http");
  EXPECT_EQ(path, "container/dir/blob");
}

TEST_F(TestAzureOptions, FromUriDisableBackgroundWrites) {
  std::string path;
  auto result = AzureOptions::FromUri("abfs://account@127.0.0.1:10000/container?background_writes=false", &path);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie().background_writes, false);
}

TEST_F(TestAzureOptions, FromUriCredentialDefault) {
  auto result =
      AzureOptions::FromUri("abfs://account.blob.core.windows.net/container?credential_kind=default", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialAnonymous) {
  auto result =
      AzureOptions::FromUri("abfs://account.blob.core.windows.net/container?credential_kind=anonymous", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialClientSecret) {
  auto result = AzureOptions::FromUri(
      "abfs://account.blob.core.windows.net/container?"
      "tenant_id=t&client_id=c&client_secret=s",
      nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialManagedIdentity) {
  auto result = AzureOptions::FromUri("abfs://account.blob.core.windows.net/container?client_id=c", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialCLI) {
  auto result = AzureOptions::FromUri("abfs://account.blob.core.windows.net/container?credential_kind=cli", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialWorkloadIdentity) {
  auto result = AzureOptions::FromUri(
      "abfs://account.blob.core.windows.net/container?credential_kind=workload_identity", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialEnvironment) {
  auto result =
      AzureOptions::FromUri("abfs://account.blob.core.windows.net/container?credential_kind=environment", nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialSASToken) {
  const std::string sas_token =
      "?se=2024-12-12T18:57:47Z&sig=pAs7qEBdI6sjUhqX1nrhNAKsTY%2B1SqLxPK%"
      "2BbAxLiopw%3D&sp=racwdxylti&spr=https,http&sr=c&sv=2024-08-04";
  auto result = AzureOptions::FromUri("abfs://file_system@account.dfs.core.windows.net/" + sas_token, nullptr);
  ASSERT_TRUE(result.ok());
}

TEST_F(TestAzureOptions, FromUriCredentialInvalid) {
  auto result = AzureOptions::FromUri(
      "abfs://file_system@account.dfs.core.windows.net/dir/file?"
      "credential_kind=invalid",
      nullptr);
  ASSERT_FALSE(result.ok());
}

TEST_F(TestAzureOptions, FromUriBlobStorageAuthority) {
  auto result = AzureOptions::FromUri(
      "abfs://account.blob.core.windows.net/container?"
      "blob_storage_authority=.blob.local",
      nullptr);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie().blob_storage_authority, ".blob.local");
}

TEST_F(TestAzureOptions, FromUriDfsStorageAuthority) {
  auto result = AzureOptions::FromUri(
      "abfs://file_system@account.dfs.core.windows.net/dir?"
      "dfs_storage_authority=.dfs.local",
      nullptr);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie().dfs_storage_authority, ".dfs.local");
}

TEST_F(TestAzureOptions, FromUriInvalidQueryParameter) {
  auto result = AzureOptions::FromUri("abfs://file_system@account.dfs.core.windows.net/dir?unknown=invalid", nullptr);
  ASSERT_FALSE(result.ok());
}

TEST_F(TestAzureOptions, MakeBlobServiceClientInvalidAccountName) {
  AzureOptions options;
  ASSERT_FALSE(options.MakeBlobServiceClient().ok());
}

TEST_F(TestAzureOptions, MakeBlobServiceClientInvalidBlobStorageScheme) {
  AzureOptions options;
  options.account_name = "user";
  options.blob_storage_scheme = "abfs";
  ASSERT_FALSE(options.MakeBlobServiceClient().ok());
}

TEST_F(TestAzureOptions, MakeDataLakeServiceClientInvalidAccountName) {
  AzureOptions options;
  ASSERT_FALSE(options.MakeDataLakeServiceClient().ok());
}

TEST_F(TestAzureOptions, MakeDataLakeServiceClientInvalidDfsStorageScheme) {
  AzureOptions options;
  options.account_name = "user";
  options.dfs_storage_scheme = "abfs";
  ASSERT_FALSE(options.MakeDataLakeServiceClient().ok());
}

}  // namespace milvus_storage::fs

namespace milvus_storage {

TEST(AzureFileSystemProducer, RejectsIamAuthenticationOverHttp) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAzure;
  config.access_key_id = "dummy-account-name";
  config.bucket_name = "dummy-container";
  config.use_iam = true;
  config.use_ssl = false;

  auto result = AzureFileSystemProducer(config).Make();
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();

  auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
  ASSERT_NE(detail, nullptr) << result.status().ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid);
  EXPECT_NE(result.status().message().find("fs.use_ssl=true"), std::string::npos);
}

}  // namespace milvus_storage
