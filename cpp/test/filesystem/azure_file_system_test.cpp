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
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <arrow/result.h>
#include <arrow/util/thread_pool.h>
#include <arrow/util/io_util.h>
#include <azure/storage/blobs.hpp>
#include <azure/storage/files/datalake.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <thread>
#include <functional>
#include <optional>
#include <stdexcept>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/azure/azurefs.h"
#include "milvus-storage/filesystem/azure/azurefs_internal.h"

namespace milvus_storage::fs {

// ============================================================================
// AzureFileSystem initialization tests (no network required)
// ============================================================================

TEST(AzureFileSystem, InitializingFilesystemWithoutAccountNameFails) {
  AzureOptions options;
  ASSERT_FALSE(options.ConfigureAccountKeyCredential("account_key").ok());

  ASSERT_TRUE(options.ConfigureClientSecretCredential("tenant_id", "client_id", "client_secret").ok());
  ASSERT_FALSE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, InvalidSharedKeyConfigurationPreservesPreviousCredential) {
  AzureOptions options;
  ASSERT_TRUE(options.ConfigureCLICredential().ok());
  const auto previous_options = options;

  const auto status = options.ConfigureAccountKeyCredential("account_key");
  ASSERT_TRUE(status.IsInvalid()) << status;
  EXPECT_TRUE(options.Equals(previous_options));
}

TEST(AzureFileSystem, InvalidSASConfigurationPreservesPreviousCredential) {
  AzureOptions options;
  ASSERT_TRUE(options.ConfigureCLICredential().ok());
  const auto previous_options = options;

  const auto status = options.ConfigureSASCredential("?sig=test-sas-secret");
  ASSERT_TRUE(status.IsInvalid()) << status;
  EXPECT_TRUE(options.Equals(previous_options));
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

TEST(AzureFileSystem, KnownSizeInputExposesNonBlockingRandomAccess) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureAnonymousCredential().ok());
  auto fs_result = AzureFileSystem::Make(options);
  ASSERT_TRUE(fs_result.ok()) << fs_result.status().ToString();
  auto azure_fs = fs_result.ValueOrDie();

  arrow::fs::FileInfo info("container/file", arrow::fs::FileType::File);
  info.set_size(42);
  auto file_result = azure_fs->OpenInputFile(info);
  ASSERT_TRUE(file_result.ok()) << file_result.status().ToString();
  auto file = file_result.ValueOrDie();
  auto* async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  auto size_result = async_file->GetSizeAsync().result();
  ASSERT_TRUE(size_result.ok()) << size_result.status().ToString();
  EXPECT_EQ(size_result.ValueOrDie(), 42);

  auto stream_result = azure_fs->OpenInputStream(info);
  ASSERT_TRUE(stream_result.ok()) << stream_result.status().ToString();
  auto stream = stream_result.ValueOrDie();
  EXPECT_NE(dynamic_cast<NonBlockingRandomAccessFile*>(stream.get()), nullptr);
}

class TestAzureFileSystem : public ::testing::Test {
  protected:
  void SetUp() override {
    AzureOptions options;
    options.account_name = "dummy-account-name";
    options.blob_storage_scheme = "http";
    options.dfs_storage_scheme = "http";
    ASSERT_TRUE(options.ConfigureManagedIdentityCredential().ok());
    auto pool_result = arrow::internal::ThreadPool::Make(1);
    ASSERT_TRUE(pool_result.ok());
    pool_ = pool_result.ValueOrDie();
    auto fs_result = AzureFileSystem::Make(options, io::IOContext(pool_.get()));
    ASSERT_TRUE(fs_result.ok()) << fs_result.status();
    fs_ = fs_result.ValueOrDie();
  }

  void SetHnsSupport(internal::HierarchicalNamespaceSupport support) {
    fs_->ForceCachedHierarchicalNamespaceSupport(static_cast<int>(support));
  }

  static void CheckAuthenticationError(const Status& status) {
    EXPECT_FALSE(status.ok());
    // Azure rejects bearer-token authentication over HTTP before any network IO.
    // Preserve that original SDK diagnosis through the Arrow boundary.
    EXPECT_NE(status.message().find("Bearer token authentication is not permitted"), std::string::npos) << status;
    const auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status;
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorAccessDenied);
    EXPECT_FALSE(detail->retryable());
  }

  std::shared_ptr<arrow::internal::ThreadPool> pool_;
  std::shared_ptr<AzureFileSystem> fs_;
};

// Exercise the real SDK response parser without an Azure account. Missing a
// required success header makes it throw std::out_of_range, not StorageException.
class AzureUploadServer {
  public:
  explicit AzureUploadServer(std::string failing_query)
      : acceptor_(context_, {boost::asio::ip::make_address_v4("127.0.0.1"), 0}),
        failing_query_(std::move(failing_query)) {
    Accept();
    worker_ = std::thread([this] { context_.run(); });
  }

  ~AzureUploadServer() {
    context_.stop();
    worker_.join();
  }

  std::string authority() const { return "127.0.0.1:" + std::to_string(acceptor_.local_endpoint().port()); }

  private:
  struct Session {
    explicit Session(boost::asio::ip::tcp::socket socket) : socket(std::move(socket)) {}
    boost::asio::ip::tcp::socket socket;
    boost::beast::flat_buffer buffer;
    boost::beast::http::request_parser<boost::beast::http::string_body> parser;
  };

  void Accept() {
    acceptor_.async_accept([this](const boost::system::error_code& error, boost::asio::ip::tcp::socket socket) {
      if (error) {
        return;
      }
      auto session = std::make_shared<Session>(std::move(socket));
      // Keep reads asynchronous so a failed test cannot block server shutdown.
      boost::beast::http::async_read_header(
          session->socket, session->buffer, session->parser,
          [this, session](const boost::system::error_code& error, size_t) {
            ASSERT_FALSE(error) << error.message();
            if (session->parser.get()[boost::beast::http::field::expect] == "100-continue") {
              boost::beast::http::response<boost::beast::http::empty_body> interim(
                  boost::beast::http::status::continue_, 11);
              boost::system::error_code write_error;
              boost::beast::http::write(session->socket, interim, write_error);
              ASSERT_FALSE(write_error) << write_error.message();
            }
            boost::beast::http::async_read(
                session->socket, session->buffer, session->parser,
                [this, session](const boost::system::error_code& error, size_t) {
                  ASSERT_FALSE(error) << error.message();
                  const auto& request = session->parser.get();
                  boost::beast::http::response<boost::beast::http::empty_body> response(
                      boost::beast::http::status::created, request.version());
                  response.set(boost::beast::http::field::etag, "\"test-etag\"");
                  response.set(boost::beast::http::field::last_modified, "Tue, 15 Sep 2026 00:00:00 GMT");
                  const auto query =
                      Azure::Core::Url("http://localhost" + std::string(request.target())).GetQueryParameters();
                  const auto comp = query.find("comp");
                  if (comp == query.end() || comp->second != failing_query_) {
                    response.set("x-ms-request-server-encrypted", "true");
                  }
                  response.keep_alive(false);
                  response.prepare_payload();
                  boost::system::error_code write_error;
                  boost::beast::http::write(session->socket, response, write_error);
                  EXPECT_FALSE(write_error) << write_error.message();
                });
          });
      Accept();
    });
  }

  boost::asio::io_context context_;
  boost::asio::ip::tcp::acceptor acceptor_;
  const std::string failing_query_;
  std::thread worker_;
};

TEST_F(TestAzureFileSystem, SynchronousReadsConvertAuthenticationExceptions) {
  auto file_result = fs_->OpenInputFile("container/file");
  ASSERT_TRUE(file_result.ok()) << file_result.status();
  const auto file = file_result.ValueOrDie();
  uint8_t out = 0;
  EXPECT_NO_THROW(CheckAuthenticationError(file->ReadAt(0, 1, &out).status()));
  EXPECT_NO_THROW(CheckAuthenticationError(file->GetSize().status()));
  EXPECT_NO_THROW(CheckAuthenticationError(file->ReadMetadata().status()));
}

TEST_F(TestAzureFileSystem, AsyncReadCompletesWithAuthenticationError) {
  arrow::fs::FileInfo info("container/file", arrow::fs::FileType::File);
  info.set_size(1);
  auto file_result = fs_->OpenInputFile(info);
  ASSERT_TRUE(file_result.ok()) << file_result.status();
  const auto file = file_result.ValueOrDie();
  auto* async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  uint8_t out = 0;
  const auto future = async_file->ReadAtAsyncInto(0, 1, &out);
  auto callback_done = arrow::Future<>::Make();
  future.AddCallback(
      [callback_done](const arrow::Result<int64_t>& result) mutable { callback_done.MarkFinished(result.status()); });
  ASSERT_TRUE(callback_done.Wait(5.0));
  CheckAuthenticationError(callback_done.status());
}

TEST_F(TestAzureFileSystem, AsyncGetSizeCompletesWithAuthenticationError) {
  auto file_result = fs_->OpenInputFile("container/file");
  ASSERT_TRUE(file_result.ok()) << file_result.status();
  const auto file = file_result.ValueOrDie();
  auto* async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  const auto future = async_file->GetSizeAsync();
  ASSERT_TRUE(future.Wait(5.0));
  CheckAuthenticationError(future.status());
}

TEST_F(TestAzureFileSystem, FilesystemOperationsConvertAuthenticationExceptions) {
  for (const auto hns :
       {internal::HierarchicalNamespaceSupport::kDisabled, internal::HierarchicalNamespaceSupport::kEnabled}) {
    SetHnsSupport(hns);
    SCOPED_TRACE(static_cast<int>(hns));
    FileSelector selector;
    selector.base_dir = "container/dir";
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->GetFileInfo("container").status()));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->GetFileInfo("container/file").status()));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->GetFileInfo(selector).status()));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->CreateDir("container/dir", true)));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->DeleteFile("container/file")));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->DeleteDir("container/dir")));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->DeleteDirContents("container/dir", false)));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->CopyFile("container/file", "container/copy")));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->OpenOutputStream("container/file").status()));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->OpenAppendStream("container/file").status()));
    EXPECT_NO_THROW(CheckAuthenticationError(fs_->OpenConditionalOutputStream("container/file", nullptr).status()));
  }
}

TEST_F(TestAzureFileSystem, AsyncUploadCountsEachFailedSdkOperationOnce) {
  struct Case {
    const char* failing_query;
    int64_t failed_count;
    int64_t write_bytes;
    int64_t failed_count_after_flush;
  };
  const Case cases[] = {{"block", 1, 0, 1}, {"blocklist", 1, 1, 2}, {"none", 0, 1, 0}};
  for (const auto& c : cases) {
    SCOPED_TRACE(c.failing_query);
    AzureUploadServer server(c.failing_query);
    AzureOptions options;
    options.account_name = "account";
    options.blob_storage_scheme = "http";
    options.blob_storage_authority = server.authority();
    options.background_writes = true;
    ASSERT_TRUE(options.ConfigureSASCredential("?sig=test-sas-secret").ok());
    auto fs_result = AzureFileSystem::Make(options, io::IOContext(pool_.get()));
    ASSERT_TRUE(fs_result.ok());
    fs_ = fs_result.ValueOrDie();
    SetHnsSupport(internal::HierarchicalNamespaceSupport::kEnabled);
    auto stream_result = fs_->OpenOutputStreamWithUploadSize("container/file", nullptr, 1);
    ASSERT_TRUE(stream_result.ok()) << stream_result.status();
    const auto stream = stream_result.ValueOrDie();
    const auto metrics = fs_->GetMetrics();
    ASSERT_NE(metrics, nullptr);
    EXPECT_EQ(metrics->GetFailedCount(), 0);
    ASSERT_TRUE(stream->Write("x", 1).ok());
    const auto closed = stream->CloseAsync();
    ASSERT_TRUE(closed.Wait(5.0)) << "Failed upload left its completion pending";
    EXPECT_EQ(metrics->GetFailedCount(), c.failed_count);
    EXPECT_EQ(metrics->GetWriteCount(), 1);
    EXPECT_EQ(metrics->GetWriteBytes(), c.write_bytes);
    if (c.failed_count != 0) {
      EXPECT_TRUE(closed.status().IsUnknownError()) << closed.status();
      EXPECT_NE(closed.status().message().find(std::string(c.failing_query) == "block" ? "StageBlock failed"
                                                                                       : "CommitBlockList failed"),
                std::string::npos)
          << closed.status();
      EXPECT_NE(closed.status().message().find("Unexpected exception:"), std::string::npos) << closed.status();
      EXPECT_NE(closed.status().message().find("/account/container/file"), std::string::npos) << closed.status();
      EXPECT_EQ(closed.status().message().find("test-sas-secret"), std::string::npos);
      EXPECT_NE(closed.status().message().find(std::string(c.failing_query) == "block" ? "bytes=1" : "block_count=1"),
                std::string::npos)
          << closed.status();
      // A failed block only propagates its status. A failed commit is attempted again.
      EXPECT_FALSE(stream->Flush().ok());
      EXPECT_EQ(metrics->GetFailedCount(), c.failed_count_after_flush);
    } else {
      EXPECT_TRUE(closed.status().ok()) << closed.status();
    }
    ASSERT_TRUE(stream->Abort().ok());
    EXPECT_EQ(metrics->GetFailedCount(), c.failed_count_after_flush);
  }
}

TEST_F(TestAzureFileSystem, FailedUploadInitializationCountsOnce) {
  SetHnsSupport(internal::HierarchicalNamespaceSupport::kEnabled);
  const auto metrics = fs_->GetMetrics();
  ASSERT_NE(metrics, nullptr);
  for (const bool conditional : {false, true}) {
    SCOPED_TRACE(conditional);
    metrics->Reset();
    const auto result = conditional ? fs_->OpenConditionalOutputStream("container/file", nullptr)
                                    : fs_->OpenOutputStream("container/file");
    ASSERT_FALSE(result.ok());
    CheckAuthenticationError(result.status());
    EXPECT_EQ(metrics->GetFailedCount(), 1);
    EXPECT_EQ(metrics->GetMultiPartUploadCreated(), 1);
    EXPECT_EQ(metrics->GetWriteBytes(), 0);
  }
}

TEST(AzureFileSystem, InvalidEndpointPortsReturnErrors) {
  AzureOptions options;
  options.account_name = "account";
  options.blob_storage_authority = "127.0.0.1:65536";
  options.dfs_storage_authority = "127.0.0.1:65536";
  ASSERT_TRUE(options.ConfigureAnonymousCredential().ok());
  EXPECT_NO_THROW(EXPECT_NE(options.MakeBlobServiceClient().status().message().find("port number is out of range"),
                            std::string::npos));
  EXPECT_NO_THROW(EXPECT_NE(options.MakeDataLakeServiceClient().status().message().find("port number is out of range"),
                            std::string::npos));
  EXPECT_NO_THROW(EXPECT_NE(AzureFileSystem::Make(options).status().message().find("port number is out of range"),
                            std::string::npos));
}

TEST(AzureFileSystem, InitializeWithClientSecretCredential) {
  AzureOptions options;
  options.account_name = "dummy-account-name";
  ASSERT_TRUE(options.ConfigureClientSecretCredential("tenant_id", "client_id", "client_secret").ok());
  ASSERT_TRUE(AzureFileSystem::Make(options).ok());
}

TEST(AzureFileSystem, CredentialFailurePreservesStateAndOriginalError) {
  AzureOptions options;
  options.account_name = "diagnostic-account";
  ASSERT_TRUE(options.ConfigureCLICredential().ok());
  const auto previous_options = options;
  const auto previous_authority = arrow::internal::GetEnvVar("AZURE_AUTHORITY_HOST");
  ASSERT_TRUE(arrow::internal::SetEnvVar("AZURE_AUTHORITY_HOST", "https://127.0.0.1:65536").ok());
  Status status;
  EXPECT_NO_THROW(status = options.ConfigureClientSecretCredential("diagnostic-tenant", "diagnostic-client",
                                                                   "do-not-log-this-secret"));
  const auto restored = previous_authority.ok()
                            ? arrow::internal::SetEnvVar("AZURE_AUTHORITY_HOST", previous_authority.ValueOrDie())
                            : arrow::internal::DelEnvVar("AZURE_AUTHORITY_HOST");
  ASSERT_TRUE(restored.ok()) << restored;
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(options.Equals(previous_options));
  for (const auto* context :
       {"diagnostic-account", "diagnostic-tenant", "diagnostic-client", "port number is out of range"}) {
    EXPECT_NE(status.message().find(context), std::string::npos) << status;
  }
  EXPECT_EQ(status.message().find("do-not-log-this-secret"), std::string::npos);
  RecordProperty("diagnostic", status.ToString());
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

class TestAzureOptions : public ::testing::Test {
  protected:
  using TokenCallback = std::function<Azure::Core::Credentials::AccessToken()>;

  static AzureOptions WithCredentialError(TokenCallback error) {
    class FailingCredential final : public Azure::Core::Credentials::TokenCredential {
   public:
      explicit FailingCredential(TokenCallback error) : TokenCredential("test"), error_(std::move(error)) {}
      Azure::Core::Credentials::AccessToken GetToken(const Azure::Core::Credentials::TokenRequestContext&,
                                                     const Azure::Core::Context&) const override {
        return error_();
      }

   private:
      const TokenCallback error_;
    };
    AzureOptions options;
    options.account_name = "account";
    options.token_credential_ = std::make_shared<FailingCredential>(std::move(error));
    return options;
  }
};

TEST_F(TestAzureOptions, ReadAndCopyPreserveSdkErrorMessagesAndClassification) {
  struct Case {
    TokenCallback error;
    const char* message;
    arrow::StatusCode status_code;
    std::optional<ExtendStatusCode> code;
  };
  const Case cases[] = {
      {[]() -> Azure::Core::Credentials::AccessToken {
         Azure::Storage::StorageException error("service failure details");
         error.StatusCode = Azure::Core::Http::HttpStatusCode::ServiceUnavailable;
         error.Message = "server diagnostic details";
         error.RequestId = "service-request-123";
         error.ClientRequestId = "client-request-123";
         throw error;
       },
       "service failure details", arrow::StatusCode::IOError, ExtendStatusCode::StorageTransientService},
      {[]() -> Azure::Core::Credentials::AccessToken {
         throw Azure::Core::Http::TransportException("network failure details");
       },
       "network failure details", arrow::StatusCode::IOError, ExtendStatusCode::StorageTransientNetwork},
      {[]() -> Azure::Core::Credentials::AccessToken {
         throw Azure::Core::Credentials::AuthenticationException("credential failure details");
       },
       "credential failure details", arrow::StatusCode::IOError, ExtendStatusCode::AwsErrorAccessDenied},
      {[]() -> Azure::Core::Credentials::AccessToken {
         throw Azure::Core::RequestFailedException("copy failed details");
       },
       "copy failed details", arrow::StatusCode::IOError, std::nullopt},
      {[]() -> Azure::Core::Credentials::AccessToken { throw std::runtime_error("unexpected SDK details"); },
       "unexpected SDK details", arrow::StatusCode::UnknownError, std::nullopt},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.message);
    const auto options = WithCredentialError(c.error);
    auto fs_result = AzureFileSystem::Make(options);
    ASSERT_TRUE(fs_result.ok());
    auto file_result = fs_result.ValueOrDie()->OpenInputFile("container/file");
    ASSERT_TRUE(file_result.ok());
    uint8_t out = 0;
    // DownloadTo and CopyFile both have explicit plain RequestFailedException paths.
    const Status statuses[] = {
        file_result.ValueOrDie()->ReadAt(0, 1, &out).status(),
        fs_result.ValueOrDie()->CopyFile("container/file", "container/copy"),
    };
    for (const auto& status : statuses) {
      ASSERT_FALSE(status.ok());
      EXPECT_EQ(status.code(), c.status_code) << status;
      EXPECT_NE(status.message().find(c.message), std::string::npos) << status;
      if (c.code == ExtendStatusCode::StorageTransientService) {
        for (const auto* field :
             {"http_status=503", "server diagnostic details", "service-request-123", "client-request-123"}) {
          EXPECT_NE(status.message().find(field), std::string::npos) << status;
        }
      }
      const auto detail = ExtendStatusDetail::UnwrapStatus(status);
      if (c.code.has_value()) {
        ASSERT_NE(detail, nullptr) << status;
        EXPECT_EQ(detail->code(), *c.code);
        EXPECT_EQ(detail->retryable(), *c.code != ExtendStatusCode::AwsErrorAccessDenied);
      } else {
        EXPECT_EQ(detail, nullptr) << status;
      }
    }
  }
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
