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
#include <initializer_list>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <arrow/result.h>
#include <arrow/util/future.h>
#include <arrow/util/io_util.h>
#include <azure/core/http/http.hpp>
#include <azure/storage/blobs.hpp>
#include <azure/storage/files/datalake.hpp>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/filesystem/azure/azurefs_internal.h"

namespace milvus_storage::fs {

TEST(AzureErrorRequestIdTest, SDKRequestId) {
  auto response =
      std::make_unique<Azure::Core::Http::RawResponse>(1, 1, Azure::Core::Http::HttpStatusCode::Forbidden, "Forbidden");
  response->SetHeader("x-ms-request-id", "header-request-id");
  response->SetHeader("authorization", "must-not-be-logged");
  Azure::Core::RequestFailedException exception(response);
  exception.RequestId = "sdk-request-id";
  exception.ErrorCode = "AuthorizationFailure";
  const auto status = internal::AzureExceptionToStatus(exception, "HeadBlob");
  EXPECT_TRUE(status.IsIOError());
  EXPECT_NE(status.message().find("request_id:sdk-request-id"), std::string::npos);
  EXPECT_EQ(status.message().find("header-request-id"), std::string::npos);
  EXPECT_EQ(status.message().find("must-not-be-logged"), std::string::npos);
  const auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorAccessDenied);
  EXPECT_NE(detail->extra_info().find("request_id:sdk-request-id"), std::string::npos);
}

TEST(AzureErrorRequestIdTest, ResponseHeadersForClassifiedAndUnclassifiedErrors) {
  for (const auto http_status :
       {Azure::Core::Http::HttpStatusCode::NotFound, Azure::Core::Http::HttpStatusCode::BadRequest}) {
    SCOPED_TRACE(static_cast<int>(http_status));
    auto response = std::make_unique<Azure::Core::Http::RawResponse>(1, 1, http_status, "error");
    response->SetHeader("X-Ms-Request-Id", "azure-header-request-id");
    response->SetHeader("x-ms-client-request-id", "client-id-not-server-id");
    Azure::Core::RequestFailedException exception(response);
    // Exercise the fallback independently of SDK header marshalling.
    exception.RequestId.clear();
    const auto status = internal::AzureExceptionToStatus(exception, "HeadBlob");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.message().find("request_id:azure-header-request-id"), std::string::npos);
    EXPECT_EQ(status.message().find("client-id-not-server-id"), std::string::npos);
    const auto detail = ExtendStatusDetail::UnwrapStatus(status);
    if (http_status == Azure::Core::Http::HttpStatusCode::NotFound) {
      ASSERT_NE(detail, nullptr);
      EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNotFound);
      EXPECT_NE(detail->extra_info().find("request_id:azure-header-request-id"), std::string::npos);
    } else {
      EXPECT_EQ(detail, nullptr);
    }
  }
}

TEST(AzureErrorRequestIdTest, MappedNotFoundPreservesErrno) {
  auto response =
      std::make_unique<Azure::Core::Http::RawResponse>(1, 1, Azure::Core::Http::HttpStatusCode::NotFound, "Not Found");
  response->SetHeader("x-ms-request-id", "head-blob-request-id");
  const Azure::Core::RequestFailedException exception(response);
  const auto not_found = arrow::Status::IOError("Path does not exist 'container/blob'")
                             .WithDetail(arrow::internal::StatusDetailFromErrno(ENOENT));
  const auto status = internal::WithAzureRequestId(not_found, exception);
  EXPECT_TRUE(status.IsIOError());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);
  EXPECT_EQ(status.message(), "Path does not exist 'container/blob' [request_id:head-blob-request-id]");
}

TEST(AzureErrorRequestIdTest, MissingRequestId) {
  Azure::Core::RequestFailedException exception("client setup failed");
  const auto status = internal::AzureExceptionToStatus(exception, "HeadBlob");
  EXPECT_EQ(status.message(), "HeadBlob Azure Error: [] client setup failed");
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
}

TEST(AzureErrorRequestIdTest, TransportErrorWithoutResponse) {
  Azure::Core::Http::TransportException exception("connection refused");
  const auto status = internal::AzureExceptionToStatus(exception, "HeadBlob");
  EXPECT_NE(status.message().find("connection refused"), std::string::npos);
  EXPECT_EQ(status.message().find("request_id:"), std::string::npos);
  const auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientNetwork);
}

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

TEST(AzureFileSystem, ReadOperationsReturnTransportErrors) {
  // Reserve a loopback port without listening so connections are refused.
  const arrow::internal::FileDescriptor socket(::socket(AF_INET, SOCK_STREAM, 0));
  ASSERT_GE(socket.fd(), 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(::bind(socket.fd(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);
  socklen_t address_size = sizeof(address);
  ASSERT_EQ(::getsockname(socket.fd(), reinterpret_cast<sockaddr*>(&address), &address_size), 0);

  AzureOptions options;
  options.account_name = "account";
  options.blob_storage_scheme = "http";
  options.blob_storage_authority = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  ASSERT_TRUE(options.ConfigureAnonymousCredential().ok());
  const auto fs_result = AzureFileSystem::Make(options);
  ASSERT_TRUE(fs_result.ok()) << fs_result.status();
  const auto file_result = fs_result.ValueOrDie()->OpenInputFile("container/blob");
  ASSERT_TRUE(file_result.ok()) << file_result.status();
  const auto file = file_result.ValueOrDie();

  for (const bool read_at : {false, true}) {
    SCOPED_TRACE(read_at ? "ReadAt" : "GetSize");
    EXPECT_NO_THROW({
      char buffer = 0;
      const auto result = read_at ? file->ReadAt(0, 1, &buffer) : file->GetSize();
      ASSERT_FALSE(result.ok());
      const auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
      ASSERT_NE(detail, nullptr) << result.status();
      EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientNetwork);
      EXPECT_TRUE(detail->retryable());
      EXPECT_FALSE(detail->extra_info().empty());
      EXPECT_NE(result.status().message().find("127.0.0.1"), std::string::npos);
    });
  }
  ASSERT_TRUE(file->Close().ok());
}

// ============================================================================
// AzureOptions::FromUri tests (no network required)
// ============================================================================

class TestAzureFileSystem {
  public:
  static void SetHierarchicalNamespaceSupport(AzureFileSystem& fs, int support) {
    fs.ForceCachedHierarchicalNamespaceSupport(support);
  }
};

class TestAzureOptions : public ::testing::Test {
  protected:
  static void ConfigureToken(AzureOptions& options,
                             std::shared_ptr<Azure::Core::Credentials::TokenCredential> credential) {
    options.token_credential_ = std::move(credential);
  }
};

namespace {
class FailingTokenCredential final : public Azure::Core::Credentials::TokenCredential {
  public:
  FailingTokenCredential() : TokenCredential("synthetic") {}
  Azure::Core::Credentials::AccessToken GetToken(const Azure::Core::Credentials::TokenRequestContext&,
                                                 const Azure::Core::Context&) const override {
    throw Azure::Core::Credentials::AuthenticationException("synthetic-secret-must-not-leak");
  }
};

void CheckAuthenticationFailure(const arrow::Status& status) {
  EXPECT_TRUE(status.IsIOError()) << status;
  EXPECT_EQ(status.message(), "Azure authentication failed");
  EXPECT_EQ(status.message().find("synthetic-secret-must-not-leak"), std::string::npos);
  // Authentication failed locally, without a remote response to classify.
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
}
}  // namespace

TEST_F(TestAzureOptions, CredentialFailureReturnsStatusForReadsAndFilesystemOperations) {
  AzureOptions options;
  options.account_name = "synthetic-account";
  ConfigureToken(options, std::make_shared<FailingTokenCredential>());
  const auto filesystem = AzureFileSystem::Make(options);
  ASSERT_TRUE(filesystem.ok()) << filesystem.status();
  const auto fs = filesystem.ValueOrDie();
  const auto opened = fs->OpenInputFile("container/blob");
  ASSERT_TRUE(opened.ok()) << opened.status();
  const auto file = opened.ValueOrDie();
  EXPECT_NO_THROW({
    CheckAuthenticationFailure(file->GetSize().status());
    char out = 0;
    CheckAuthenticationFailure(file->ReadAt(0, 1, &out).status());
    CheckAuthenticationFailure(file->ReadMetadata().status());
    CheckAuthenticationFailure(fs->GetFileInfo("container").status());
    CheckAuthenticationFailure(fs->DeleteDir("container"));
    CheckAuthenticationFailure(fs->Move("container", "other-container"));
    arrow::fs::FileSelector select;
    CheckAuthenticationFailure(fs->GetFileInfo(select).status());
    // Probe unknown, flat and hierarchical namespace operation dispatch.
    for (int hns : {0, 2, 3}) {
      TestAzureFileSystem::SetHierarchicalNamespaceSupport(*fs, hns);
      CheckAuthenticationFailure(fs->GetFileInfo("container/blob").status());
      CheckAuthenticationFailure(fs->CreateDir("container/dir", true));
      CheckAuthenticationFailure(fs->DeleteFile("container/blob"));
      CheckAuthenticationFailure(fs->DeleteDirContents("container", false));
      if (hns != 2)
        CheckAuthenticationFailure(fs->Move("container/a", "container/b"));
      CheckAuthenticationFailure(fs->CopyFile("container/a", "container/b"));
      CheckAuthenticationFailure(fs->OpenOutputStream("container/blob", nullptr).status());
      CheckAuthenticationFailure(fs->OpenAppendStream("container/blob", nullptr).status());
      CheckAuthenticationFailure(fs->OpenConditionalOutputStream("container/blob", nullptr).status());
    }
  });
  ASSERT_TRUE(file->Close().ok());
}

TEST_F(TestAzureOptions, CredentialFailureCompletesAsyncReadWithStatus) {
  AzureOptions options;
  options.account_name = "synthetic-account";
  ConfigureToken(options, std::make_shared<FailingTokenCredential>());
  const auto filesystem = AzureFileSystem::Make(options);
  ASSERT_TRUE(filesystem.ok()) << filesystem.status();
  const auto opened = filesystem.ValueOrDie()->OpenInputFile("container/blob");
  ASSERT_TRUE(opened.ok()) << opened.status();
  const auto file = opened.ValueOrDie();
  // Actual Arrow executor -> Blob SDK -> raw TokenCredential failure. An
  // escaped auth exception terminates the worker instead of completing this Future.
  EXPECT_NO_THROW(CheckAuthenticationFailure(file->ReadAsync(0, 1).result().status()));
  ASSERT_TRUE(file->Close().ok());
}

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
