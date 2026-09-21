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

#include <arrow/testing/gtest_util.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <sstream>
#include <thread>
#include <type_traits>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/s3/model/PutObjectResult.h>

#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/filesystem/upload_sizable.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_client_builder.h"
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "milvus-storage/filesystem/fs.h"

#include "test_env.h"

namespace milvus_storage {
// Internal factory shared by ordinary S3 and CRT reads, defined in s3_filesystem.cpp.
Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, int64_t nbytes);

// ============================================================================
// Non-cloud unit tests — S3 SDK initialized but no real cloud connection needed
// ============================================================================

class S3UnitTest : public ::testing::Test {
  protected:
  void SetUp() override {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      GTEST_SKIP() << "S3 unit tests only run for AWS provider";
    }
  }
  static void SetUpTestSuite() {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      return;
    }
    ASSERT_TRUE(EnsureS3Initialized().ok());
  }
};

// Exercise the real SDK HTTP/error parser, including responses larger than the
// caller's read buffer. Async server operations let teardown stop a failed test.
class S3ReadResponseServer {
  using Tcp = boost::asio::ip::tcp;
  using Response = boost::beast::http::response<boost::beast::http::string_body>;

  public:
  explicit S3ReadResponseServer(std::vector<Response> responses)
      : acceptor_(context_, {boost::asio::ip::address_v4::loopback(), 0}), responses_(std::move(responses)) {
    Accept();
    thread_ = std::thread([this] { context_.run(); });
  }

  ~S3ReadResponseServer() {
    context_.stop();
    thread_.join();
  }

  std::string endpoint() const { return "127.0.0.1:" + std::to_string(acceptor_.local_endpoint().port()); }
  size_t requests() const { return requests_.load(); }
  bool only_gets() const { return only_gets_.load(); }

  private:
  void Accept() {
    socket_ = std::make_unique<Tcp::socket>(context_);
    acceptor_.async_accept(*socket_, [this](const boost::system::error_code& error) {
      if (error) {
        return;
      }
      request_ = {};
      buffer_.consume(buffer_.size());
      boost::beast::http::async_read(
          *socket_, buffer_, request_, [this](const boost::system::error_code& read_error, size_t) {
            if (read_error) {
              return;
            }
            if (request_.method() != boost::beast::http::verb::get) {
              only_gets_ = false;
            }
            const auto index = requests_.fetch_add(1);
            auto& response = responses_[std::min(index, responses_.size() - 1)];
            response.keep_alive(false);
            response.prepare_payload();
            boost::beast::http::async_write(*socket_, response, [this](const boost::system::error_code&, size_t) {
              boost::system::error_code ignored;
              socket_->close(ignored);
              Accept();
            });
          });
    });
  }

  boost::asio::io_context context_;
  Tcp::acceptor acceptor_;
  std::unique_ptr<Tcp::socket> socket_;
  boost::beast::flat_buffer buffer_;
  boost::beast::http::request<boost::beast::http::string_body> request_;
  std::vector<Response> responses_;
  std::atomic<size_t> requests_{0};
  std::atomic<bool> only_gets_{true};
  std::thread thread_;
};

class S3ReadResponseTest : public S3UnitTest, public ::testing::WithParamInterface<bool> {
  protected:
  static void SetUpTestSuite() {
    S3UnitTest::SetUpTestSuite();
    std::atexit([] { (void)EnsureS3Finalized(); });
  }
};

TEST_F(S3UnitTest, ResponseStreamWritesDirectlyIntoCallerBuffer) {
  std::array<char, 10> guarded;
  guarded.fill('?');
  Aws::Utils::Stream::ResponseStream response(AwsWriteableStreamFactory(guarded.data() + 1, 8));
  auto& stream = response.GetUnderlyingStream();
  stream.write("ab", 2);
  ASSERT_TRUE(stream.good());
  EXPECT_EQ(std::string(guarded.data() + 1, 2), "ab");
  EXPECT_EQ(stream.tellp(), std::streampos(2));
  stream.write("cdefgh", 6);
  EXPECT_EQ(std::string(guarded.data() + 1, 8), "abcdefgh");
  EXPECT_EQ(stream.tellp(), std::streampos(8));
  stream.seekg(0);
  EXPECT_EQ(stream.get(), 'a');
  stream.unget();
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), "abcdefgh");
  EXPECT_EQ(guarded.front(), '?');
  EXPECT_EQ(guarded.back(), '?');
}

TEST_F(S3UnitTest, ResponseStreamGrowsWithoutOverrunningCallerBuffer) {
  std::array<char, 6> guarded;
  guarded.fill('?');
  Aws::Utils::Stream::ResponseStream response(AwsWriteableStreamFactory(guarded.data() + 1, 4));
  auto& stream = response.GetUnderlyingStream();
  stream.write("ab", 2);
  EXPECT_EQ(stream.get(), 'a');
  EXPECT_EQ(stream.rdbuf()->in_avail(), 1);
  stream.write("cdef", 4);
  ASSERT_TRUE(stream.good());
  EXPECT_EQ(stream.tellp(), std::streampos(6));
  EXPECT_EQ(stream.tellg(), std::streampos(1));
  EXPECT_EQ(stream.rdbuf()->in_avail(), 5);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), "bcdef");
  stream.seekg(0);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), "abcdef");
  stream.seekg(0);
  stream.seekp(0);
  stream.write("err", 3);
  EXPECT_EQ(stream.tellp(), std::streampos(3));
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), "err");
  EXPECT_EQ(guarded.front(), '?');
  EXPECT_EQ(guarded.back(), '?');
}

TEST_F(S3UnitTest, ResponseStreamRewindHidesPreviousResponseTail) {
  std::array<char, 64> data;
  data.fill('?');
  Aws::Utils::Stream::ResponseStream response(AwsWriteableStreamFactory(data.data(), data.size()));
  auto& stream = response.GetUnderlyingStream();
  stream.write("previous successful object bytes", 32);
  stream.seekg(0);
  stream.seekp(0);
  stream.write("<Error/>", 8);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(stream), {}), "<Error/>");
  stream.seekg(9);
  EXPECT_TRUE(stream.fail());
  stream.clear();
  stream.seekp(-1);
  EXPECT_TRUE(stream.fail());
  stream.clear();
  stream.seekg(8);
  stream.seekp(0);
  stream.unget();
  EXPECT_TRUE(stream.fail());
}

TEST_P(S3ReadResponseTest, PreservesErrorsLargerThanReadBuffer) {
  namespace http = boost::beast::http;
  for (const auto& [http_status, code] :
       std::vector<std::pair<http::status, std::string>>{{http::status::not_found, "NoSuchKey"},
                                                         {http::status::forbidden, "AccessDenied"},
                                                         {http::status::service_unavailable, "SlowDown"}}) {
    SCOPED_TRACE(code);
    http::response<http::string_body> response{http_status, 11};
    response.set(http::field::content_type, "application/xml");
    response.body() = "<?xml version=\"1.0\"?><Error><Code>" + code +
                      "</Code><Message>original service diagnostic must survive a small read</Message></Error>";
    S3ReadResponseServer server({std::move(response)});
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = server.endpoint();
    options.use_crt_async_reads = GetParam();
    options.connect_timeout = 2;
    options.request_timeout = 5;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(0);
    ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
    ASSERT_AND_ASSIGN(auto input, fs->OpenInputFile("bucket/object"));
    std::array<uint8_t, 16> data;
    data.fill(0xa5);
    const auto result = input->ReadAt(0, data.size(), data.data());
    ASSERT_FALSE(result.ok());
    // Native CRT turns exhausted 503 retries into its own throttling error
    // without exposing the server XML. Preserve that SDK diagnostic as well.
    const auto* diagnostic = GetParam() && http_status == http::status::service_unavailable
                                 ? "Response code indicates throttling"
                                 : "original service diagnostic";
    EXPECT_NE(result.status().message().find(diagnostic), std::string::npos);
    if (http_status == http::status::not_found) {
      const auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
      ASSERT_NE(detail, nullptr);
      EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNotFound);
    } else {
      const auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
      ASSERT_NE(detail, nullptr);
      EXPECT_EQ(detail->code(), http_status == http::status::forbidden ? ExtendStatusCode::AwsErrorAccessDenied
                                                                       : ExtendStatusCode::StorageTransientThrottling);
    }
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](uint8_t value) { return value == 0xa5; }));
    EXPECT_TRUE(server.only_gets());
    ASSERT_STATUS_OK(input->Close());
  }
}

TEST_P(S3ReadResponseTest, RetriesErrorThenReadsIntoCallerBuffer) {
  namespace http = boost::beast::http;
  http::response<http::string_body> error{http::status::service_unavailable, 11};
  error.set(http::field::content_type, "application/xml");
  error.body() = "<?xml version=\"1.0\"?><Error><Code>SlowDown</Code><Message>Please retry this read</Message></Error>";
  http::response<http::string_body> success{http::status::partial_content, 11};
  success.set(http::field::etag, "\"4032af8d61035123906e58e067140cc5\"");
  success.set(http::field::content_range, "bytes 0-15/16");
  success.body() = "0123456789abcdef";
  S3ReadResponseServer server({std::move(error), std::move(success)});
  auto options = S3Options::FromAccessKey("ak", "sk");
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = server.endpoint();
  options.use_crt_async_reads = GetParam();
  options.connect_timeout = 2;
  options.request_timeout = 5;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(1);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto input, fs->OpenInputFile("bucket/object"));
  std::array<char, 16> data{};
  ASSERT_AND_ASSIGN(const auto bytes_read, input->ReadAt(0, data.size(), data.data()));
  EXPECT_EQ(bytes_read, data.size());
  EXPECT_EQ(std::string(data.data(), data.size()), "0123456789abcdef");
  EXPECT_EQ(server.requests(), 2);
  EXPECT_TRUE(server.only_gets());
  ASSERT_STATUS_OK(input->Close());
}

TEST_P(S3ReadResponseTest, ReadsMultiplePartsAndPreservesLaterErrors) {
  if (!GetParam()) {
    GTEST_SKIP() << "Only native CRT splits a range into multiple requests";
  }
  namespace http = boost::beast::http;
  constexpr size_t part_size = 8 * 1024 * 1024;
  constexpr size_t total_size = part_size + 16;
  for (const bool fail_last_part : {false, true}) {
    SCOPED_TRACE(fail_last_part);
    http::response<http::string_body> first{http::status::partial_content, 11};
    first.set(http::field::etag, "\"4032af8d61035123906e58e067140cc5\"");
    first.set(http::field::content_range,
              "bytes 0-" + std::to_string(part_size - 1) + "/" + std::to_string(total_size));
    first.body() = std::string(part_size, 'a');
    http::response<http::string_body> last{fail_last_part ? http::status::not_found : http::status::partial_content,
                                           11};
    if (fail_last_part) {
      last.set(http::field::content_type, "application/xml");
      last.body() =
          "<?xml version=\"1.0\"?><Error><Code>NoSuchKey</Code><Message>object removed between parts</Message></Error>";
    } else {
      last.set(http::field::etag, "\"4032af8d61035123906e58e067140cc5\"");
      last.set(http::field::content_range, "bytes " + std::to_string(part_size) + "-" + std::to_string(total_size - 1) +
                                               "/" + std::to_string(total_size));
      last.body() = std::string(16, 'b');
    }
    S3ReadResponseServer server({std::move(first), std::move(last)});
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = server.endpoint();
    options.use_crt_async_reads = true;
    options.connect_timeout = 2;
    options.request_timeout = 5;
    ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
    ASSERT_AND_ASSIGN(auto input, fs->OpenInputFile("bucket/object"));
    std::vector<uint8_t> guarded(total_size + 2, 0xa5);
    auto* const data = guarded.data() + 1;
    const auto result = input->ReadAt(0, total_size, data);
    if (fail_last_part) {
      ASSERT_FALSE(result.ok());
      const auto detail = ExtendStatusDetail::UnwrapStatus(result.status());
      ASSERT_NE(detail, nullptr);
      EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNotFound);
      EXPECT_NE(result.status().message().find("object removed between parts"), std::string::npos);
    } else {
      ASSERT_OK(result.status());
      EXPECT_EQ(*result, total_size);
    }
    // Failed reads do not promise valid output, but must never overrun it.
    if (!fail_last_part) {
      EXPECT_TRUE(std::all_of(data, data + part_size, [](uint8_t value) { return value == 'a'; }));
      EXPECT_TRUE(std::all_of(data + part_size, data + total_size, [](uint8_t value) { return value == 'b'; }));
    }
    EXPECT_EQ(guarded.front(), 0xa5);
    EXPECT_EQ(guarded.back(), 0xa5);
    EXPECT_EQ(server.requests(), 2);
    EXPECT_TRUE(server.only_gets());
    ASSERT_STATUS_OK(input->Close());
  }
}

#ifdef WITH_CRT
INSTANTIATE_TEST_SUITE_P(SdkAndCrt, S3ReadResponseTest, ::testing::Bool());
#else
INSTANTIATE_TEST_SUITE_P(Sdk, S3ReadResponseTest, ::testing::Values(false));
#endif

TEST_F(S3UnitTest, TestExtendErrorInFs) {
  Aws::Client::AWSError<Aws::S3::S3Errors> test_err(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                    Aws::Client::RetryableType::NOT_RETRYABLE, "AwsErrorNoSuchUpload",
                                                    "Just for test");

  auto status = fs::internal::ErrorToStatus("test", test_err);
  ASSERT_STATUS_NOT_OK(status);
  auto extend_status = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(extend_status, nullptr);
  ASSERT_EQ(extend_status->code(), ExtendStatusCode::AwsErrorNoSuchUpload);
  ASSERT_TRUE(status.ToString().find(extend_status->ToString()) != std::string::npos);
}

TEST_F(S3UnitTest, TestErrorToStatusPermanentVsTransient) {
  // NoSuchKey: permanent, tagged AwsErrorNotFound.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::NO_SUCH_KEY, Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchKey", "object gone");
    auto status = fs::internal::ErrorToStatus("test", error);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNotFound);
  }
  // AccessDenied: permanent, tagged AwsErrorAccessDenied.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    auto status = fs::internal::ErrorToStatus("test", error);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorAccessDenied);
  }
  // A recognized error type the SDK judged non-retryable: tagged permanent.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::VALIDATION, Aws::Client::RetryableType::NOT_RETRYABLE, "ValidationError", "bad request");
    auto status = fs::internal::ErrorToStatus("test", error);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNonRetryable);
  }
  // MinIO-style SlowDown: arrives as UNKNOWN + non-retryable, but it is a
  // genuine transient (rate limiting), so it must carry retryable throttling
  // detail instead of being tagged permanent.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "rate limited");
    auto status = fs::internal::ErrorToStatus("test", error);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
    EXPECT_TRUE(detail->retryable());
  }
  // Recognized retryable transient: explicit retryable throttling detail.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::SLOW_DOWN, Aws::Client::RetryableType::RETRYABLE_THROTTLING, "SlowDown", "rate limited");
    auto status = fs::internal::ErrorToStatus("test", error);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
    EXPECT_TRUE(detail->retryable());
  }
}

TEST_F(S3UnitTest, TestSignRequest) {
  // GET
  {
    Aws::Http::URI uri("https://storage.googleapis.com/my-bucket/my-object");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_GET);

    bool ok = auth_signer::googv4::SignRequest(request, "GOOGACCESSKEY", "SECRET");
    ASSERT_TRUE(ok);
    EXPECT_TRUE(request->HasHeader("Authorization"));
    auto auth = request->GetHeaderValue("Authorization");
    EXPECT_NE(std::string(auth).find("GOOG4-HMAC-SHA256"), std::string::npos);
    EXPECT_TRUE(request->HasHeader("x-goog-date"));
    EXPECT_TRUE(request->HasHeader("x-goog-content-sha256"));
  }

  // POST
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_POST);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    EXPECT_NE(std::string(request->GetHeaderValue("Authorization")).find("GOOG4-HMAC-SHA256"), std::string::npos);
  }

  // PUT
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_PUT);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    EXPECT_NE(std::string(request->GetHeaderValue("Authorization")).find("GOOG4-HMAC-SHA256"), std::string::npos);
  }

  // DELETE
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_DELETE);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    EXPECT_NE(std::string(request->GetHeaderValue("Authorization")).find("GOOG4-HMAC-SHA256"), std::string::npos);
  }

  // HEAD
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_HEAD);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    EXPECT_NE(std::string(request->GetHeaderValue("Authorization")).find("GOOG4-HMAC-SHA256"), std::string::npos);
  }

  // Empty body uses empty SHA256
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_GET);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    auto content_sha = std::string(request->GetHeaderValue("x-goog-content-sha256"));
    EXPECT_EQ(content_sha, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  }

  // With body stream
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_PUT);
    auto body = Aws::MakeShared<Aws::StringStream>("test");
    (*body) << "hello world";
    request->AddContentBody(body);

    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    auto content_sha = std::string(request->GetHeaderValue("x-goog-content-sha256"));
    EXPECT_NE(content_sha, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_FALSE(content_sha.empty());
  }

  // With query params
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key?param_b=2&param_a=1");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_GET);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    EXPECT_TRUE(request->HasHeader("Authorization"));
  }

  // With multiple headers — verify SignedHeaders present
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_GET);
    request->SetHeaderValue("x-custom-header", "value1");
    request->SetHeaderValue("x-another-header", "value2");

    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "AK", "SK"));
    auto auth = std::string(request->GetHeaderValue("Authorization"));
    EXPECT_NE(auth.find("SignedHeaders="), std::string::npos);
  }

  // Credential scope format
  {
    Aws::Http::URI uri("https://storage.googleapis.com/bucket/key");
    auto request =
        Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("test", uri, Aws::Http::HttpMethod::HTTP_GET);
    ASSERT_TRUE(auth_signer::googv4::SignRequest(request, "MYACCESSKEY", "MYSECRETKEY"));
    auto auth = std::string(request->GetHeaderValue("Authorization"));
    EXPECT_NE(auth.find("Credential=MYACCESSKEY/"), std::string::npos);
    EXPECT_NE(auth.find("/auto/storage/goog4_request"), std::string::npos);
    EXPECT_NE(auth.find("Signature="), std::string::npos);
  }
}

TEST_F(S3UnitTest, TestS3Options) {
  // Defaults
  {
    auto options = S3Options::Defaults();
    EXPECT_EQ(options.credentials_kind, S3CredentialsKind::Default);
    EXPECT_NE(options.credentials_provider, nullptr);
  }

  // Anonymous
  {
    auto options = S3Options::Anonymous();
    EXPECT_EQ(options.credentials_kind, S3CredentialsKind::Anonymous);
    EXPECT_NE(options.credentials_provider, nullptr);
  }

  // FromAccessKey with token
  {
    auto options = S3Options::FromAccessKey("myak", "mysk", "mytoken");
    EXPECT_EQ(options.credentials_kind, S3CredentialsKind::Explicit);
    EXPECT_EQ(options.GetAccessKey(), "myak");
    EXPECT_EQ(options.GetSecretKey(), "mysk");
    EXPECT_EQ(options.GetSessionToken(), "mytoken");
  }

  // FromAccessKey without token
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    EXPECT_EQ(options.credentials_kind, S3CredentialsKind::Explicit);
    EXPECT_EQ(options.GetAccessKey(), "ak");
    EXPECT_EQ(options.GetSecretKey(), "sk");
    EXPECT_EQ(options.GetSessionToken(), "");
  }

  // FromUri — bucket and path
  {
    std::string out_path;
    auto result = S3Options::FromUri("s3://mybucket/some/path?region=us-east-1", &out_path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(out_path, "mybucket/some/path");
  }

  // FromUri — bucket only
  {
    std::string out_path;
    auto result = S3Options::FromUri("s3://mybucket?region=us-east-1", &out_path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(out_path, "mybucket");
  }

  // FromUri — empty
  {
    std::string out_path;
    auto result = S3Options::FromUri("s3://", &out_path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(out_path, "");
  }

  // FromUri — query params (region + scheme)
  {
    std::string out_path;
    auto result = S3Options::FromUri("s3://mybucket/path?region=us-west-2&scheme=http", &out_path);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->region, "us-west-2");
    EXPECT_EQ(result->scheme, "http");
  }

  // FromUri — endpoint_override
  {
    auto result = S3Options::FromUri("s3://mybucket?endpoint_override=localhost:9000&region=us-east-1");
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->endpoint_override, "localhost:9000");
  }

  // FromUri — allow_bucket_creation
  {
    auto result = S3Options::FromUri("s3://mybucket?allow_bucket_creation=true&region=us-east-1");
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->allow_bucket_creation);
  }

  // FromUri — bad param
  {
    auto result = S3Options::FromUri("s3://mybucket?bad_param=x&region=us-east-1");
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.status().ToString().find("Unexpected query parameter"), std::string::npos);
  }

  // FromUri — credentials in URI
  {
    auto result = S3Options::FromUri("s3://user:pass@mybucket/path?region=us-east-1");
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetAccessKey(), "user");
    EXPECT_EQ(result->GetSecretKey(), "pass");
    EXPECT_EQ(result->credentials_kind, S3CredentialsKind::Explicit);
  }

  // Equals
  {
    auto opt1 = S3Options::FromAccessKey("ak", "sk");
    auto opt2 = S3Options::FromAccessKey("ak", "sk");
    EXPECT_TRUE(opt1.Equals(opt2));

    auto opt3 = S3Options::FromAccessKey("ak", "sk2");
    EXPECT_FALSE(opt1.Equals(opt3));
  }

  // S3ProxyOptions::Equals
  {
    S3ProxyOptions p1;
    p1.scheme = "http";
    p1.host = "proxy.example.com";
    p1.port = 8080;
    p1.username = "user";
    p1.password = "pass";

    S3ProxyOptions p2 = p1;
    EXPECT_TRUE(p1.Equals(p2));

    p2.port = 9090;
    EXPECT_FALSE(p1.Equals(p2));
  }

  // ResolveS3BucketRegion
  {
    EXPECT_FALSE(ResolveS3BucketRegion("").ok());
    EXPECT_FALSE(ResolveS3BucketRegion("valid-bucket").ok());
  }
}

TEST_F(S3UnitTest, TestS3RetryStrategy) {
  // Default strategy
  {
    auto strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(3);
    ASSERT_NE(strategy, nullptr);
  }

  // Standard strategy
  {
    auto strategy = S3RetryStrategy::GetAwsStandardRetryStrategy(3);
    ASSERT_NE(strategy, nullptr);
  }

  // ShouldRetry + CalculateDelay
  {
    auto strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(3);
    S3RetryStrategy::AWSErrorDetail detail;
    detail.error_type = static_cast<int>(Aws::Client::CoreErrors::NETWORK_CONNECTION);
    detail.message = "Connection reset";
    detail.exception_name = "NetworkConnection";
    detail.should_retry = true;

    EXPECT_TRUE(strategy->ShouldRetry(detail, 0));
    EXPECT_GE(strategy->CalculateDelayBeforeNextRetry(detail, 0), 0);
  }
}

TEST_F(S3UnitTest, TestDetectS3Backend) {
  {
    Aws::Http::HeaderValueCollection headers;
    headers["server"] = "AmazonS3";
    EXPECT_EQ(fs::internal::DetectS3Backend(headers), fs::internal::S3Backend::Amazon);
  }
  {
    Aws::Http::HeaderValueCollection headers;
    headers["server"] = "MinIO";
    EXPECT_EQ(fs::internal::DetectS3Backend(headers), fs::internal::S3Backend::Minio);
  }
  {
    Aws::Http::HeaderValueCollection headers;
    headers["server"] = "SomeOtherServer";
    EXPECT_EQ(fs::internal::DetectS3Backend(headers), fs::internal::S3Backend::Other);
  }
  {
    Aws::Http::HeaderValueCollection headers;
    EXPECT_EQ(fs::internal::DetectS3Backend(headers), fs::internal::S3Backend::Other);
  }
}

TEST_F(S3UnitTest, TestIsConnectError) {
  // Retryable network error
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::NETWORK_CONNECTION, true);
    EXPECT_TRUE(fs::internal::IsConnectError(error));
  }
  // SlowDown
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(
        Aws::Client::CoreErrors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "rate limited");
    EXPECT_TRUE(fs::internal::IsConnectError(error));
  }
  // SlowDownWrite
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(
        Aws::Client::CoreErrors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDownWrite", "rate limited");
    EXPECT_TRUE(fs::internal::IsConnectError(error));
  }
  // XMinioServerNotInitialized
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::UNKNOWN,
                                                         Aws::Client::RetryableType::NOT_RETRYABLE,
                                                         "XMinioServerNotInitialized", "Server not initialized");
    EXPECT_TRUE(fs::internal::IsConnectError(error));
  }
  // Non-retryable access denied
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(
        Aws::Client::CoreErrors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    EXPECT_FALSE(fs::internal::IsConnectError(error));
  }
}

TEST_F(S3UnitTest, TestS3ErrorClassification) {
  // IsNotFound — bucket
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::NO_SUCH_BUCKET, Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchBucket", "not found");
    EXPECT_TRUE(fs::internal::IsNotFound(error));
  }
  // IsNotFound — resource
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::RESOURCE_NOT_FOUND,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "ResourceNotFound",
                                                   "not found");
    EXPECT_TRUE(fs::internal::IsNotFound(error));
  }
  // IsNotFound — false
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    EXPECT_FALSE(fs::internal::IsNotFound(error));
  }
  // IsAlreadyExists — BUCKET_ALREADY_EXISTS
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::BUCKET_ALREADY_EXISTS,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "BucketAlreadyExists",
                                                   "already exists");
    EXPECT_TRUE(fs::internal::IsAlreadyExists(error));
  }
  // IsAlreadyExists — BUCKET_ALREADY_OWNED_BY_YOU
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::BUCKET_ALREADY_OWNED_BY_YOU,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "BucketAlreadyOwnedByYou",
                                                   "owned");
    EXPECT_TRUE(fs::internal::IsAlreadyExists(error));
  }
}

TEST_F(S3UnitTest, TestS3ErrorToString) {
  EXPECT_EQ(fs::internal::S3ErrorToString(Aws::S3::S3Errors::NO_SUCH_BUCKET), "NO_SUCH_BUCKET");
  EXPECT_EQ(fs::internal::S3ErrorToString(Aws::S3::S3Errors::NO_SUCH_KEY), "NO_SUCH_KEY");
  EXPECT_EQ(fs::internal::S3ErrorToString(Aws::S3::S3Errors::ACCESS_DENIED), "ACCESS_DENIED");
  EXPECT_EQ(fs::internal::S3ErrorToString(Aws::S3::S3Errors::BUCKET_ALREADY_EXISTS), "BUCKET_ALREADY_EXISTS");

  // Unknown error code
  {
    auto unknown_error = static_cast<Aws::S3::S3Errors>(9999);
    auto result = fs::internal::S3ErrorToString(unknown_error);
    EXPECT_NE(result.find("[code "), std::string::npos);
  }
}

TEST_F(S3UnitTest, TestErrorToStatus) {
  auto AssertRetryableCode = [](const arrow::Status& status, ExtendStatusCode expected_code) {
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), expected_code);
    EXPECT_TRUE(detail->retryable());
  };

  auto AssertNonRetryable = [](const arrow::Status& status) {
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    if (detail) {
      EXPECT_FALSE(detail->retryable());
    }
  };

  // NO_SUCH_UPLOAD
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchUpload",
                                                   "Upload not found");
    auto status = fs::internal::ErrorToStatus("test_prefix", "CompleteMultipart", error);
    AssertRetryableCode(status, ExtendStatusCode::AwsErrorNoSuchUpload);
  }

  // AWS SDK retryable error
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::INTERNAL_FAILURE, Aws::Client::RetryableType::RETRYABLE, "InternalFailure", "retryable");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientNetwork);
  }

  // NETWORK_CONNECTION
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NETWORK_CONNECTION,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NetworkConnection",
                                                   "network");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientNetwork);
  }

  // REQUEST_TIMEOUT
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::REQUEST_TIMEOUT, Aws::Client::RetryableType::NOT_RETRYABLE, "RequestTimeout", "timeout");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientTimeout);
  }

  // HTTP 408
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "RequestTimeout", "timeout");
    error.SetResponseCode(Aws::Http::HttpResponseCode::REQUEST_TIMEOUT);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientTimeout);
  }

  // HTTP 429 from S3-compatible backends may arrive as UNKNOWN.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "TooManyRequests", "rate limited");
    error.SetResponseCode(Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // SLOW_DOWN
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::SLOW_DOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // THROTTLING
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::THROTTLING, Aws::Client::RetryableType::NOT_RETRYABLE, "Throttling", "throttled");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // MinIO SlowDown
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // MinIO SlowDownWrite
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDownWrite", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // SERVICE_UNAVAILABLE
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::SERVICE_UNAVAILABLE,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "ServiceUnavailable",
                                                   "unavailable");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 500
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "InternalError", "internal");
    error.SetResponseCode(Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 502
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "BadGateway", "bad gateway");
    error.SetResponseCode(Aws::Http::HttpResponseCode::BAD_GATEWAY);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 503
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "ServiceUnavailable", "unavailable");
    error.SetResponseCode(Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 504
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "GatewayTimeout", "gateway timeout");
    error.SetResponseCode(Aws::Http::HttpResponseCode::GATEWAY_TIMEOUT);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // XMinioServerNotInitialized
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE,
                                                   "XMinioServerNotInitialized", "server not initialized");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // PRECONDITION_FAILED
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "PreconditionFailed",
                                                   "condition failed");
    error.SetResponseCode(Aws::Http::HttpResponseCode::PRECONDITION_FAILED);
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorPreConditionFailed);
    EXPECT_FALSE(detail->retryable());
  }

  // CONFLICT
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "Conflict", "conflict");
    error.SetResponseCode(Aws::Http::HttpResponseCode::CONFLICT);
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error);
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorConflict);
    EXPECT_FALSE(detail->retryable());
  }

  // Generic UNKNOWN IOError with no recognized permanent or transient signal
  // remains plain IOError with no ExtendStatus detail.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SomeBackendSpecificError", "opaque");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    AssertNonRetryable(status);
    EXPECT_TRUE(status.IsIOError());
    EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  }
}

TEST_F(S3UnitTest, TestOutcomeToStatus) {
  // Success
  {
    Aws::S3::Model::PutObjectResult put_result;
    Aws::Utils::Outcome<Aws::S3::Model::PutObjectResult, Aws::Client::AWSError<Aws::S3::S3Errors>> outcome(
        std::move(put_result));
    EXPECT_TRUE(fs::internal::OutcomeToStatus("prefix", "PutObject", outcome).ok());
  }

  // Failure
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    Aws::Utils::Outcome<Aws::S3::Model::PutObjectResult, Aws::Client::AWSError<Aws::S3::S3Errors>> outcome(
        std::move(error));
    EXPECT_FALSE(fs::internal::OutcomeToStatus("prefix", "PutObject", outcome).ok());
  }
}

TEST_F(S3UnitTest, TestConnectRetryStrategy) {
  // ShouldRetry — retryable vs non-retryable
  {
    fs::internal::ConnectRetryStrategy strategy(200, 6000);

    Aws::Client::AWSError<Aws::Client::CoreErrors> retryable_error(Aws::Client::CoreErrors::NETWORK_CONNECTION, true);
    EXPECT_TRUE(strategy.ShouldRetry(retryable_error, 0));

    Aws::Client::AWSError<Aws::Client::CoreErrors> non_retryable_error(
        Aws::Client::CoreErrors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    EXPECT_FALSE(strategy.ShouldRetry(non_retryable_error, 0));
  }

  // Max duration boundary
  {
    fs::internal::ConnectRetryStrategy strategy(200, 1000);
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::NETWORK_CONNECTION, true);
    EXPECT_FALSE(strategy.ShouldRetry(error, 5));  // 5 * 200ms = 1000ms = max
    EXPECT_TRUE(strategy.ShouldRetry(error, 4));   // 4 * 200ms = 800ms < max
  }

  // CalculateDelay is constant
  {
    fs::internal::ConnectRetryStrategy strategy(300, 6000);
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(Aws::Client::CoreErrors::NETWORK_CONNECTION, true);
    EXPECT_EQ(strategy.CalculateDelayBeforeNextRetry(error, 0), 300);
    EXPECT_EQ(strategy.CalculateDelayBeforeNextRetry(error, 5), 300);
  }
}

TEST_F(S3UnitTest, TestAwsStringConversion) {
  std::string original = "hello-world";
  auto aws_str = fs::internal::ToAwsString(original);
  EXPECT_EQ(fs::internal::FromAwsString(aws_str), original);
}

TEST_F(S3UnitTest, TestPathUtilities) {
  // DetectAbsolutePath
  {
    EXPECT_TRUE(arrow::fs::internal::DetectAbsolutePath("/foo/bar"));
    EXPECT_TRUE(arrow::fs::internal::DetectAbsolutePath("/"));
    EXPECT_FALSE(arrow::fs::internal::DetectAbsolutePath("foo/bar"));
    EXPECT_FALSE(arrow::fs::internal::DetectAbsolutePath(""));
  }

  // PathNotFound
  {
    auto status = arrow::fs::internal::PathNotFound("/missing/path");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.ToString().find("/missing/path"), std::string::npos);
  }

  // IsADir
  {
    auto status = arrow::fs::internal::IsADir("/some/dir");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.ToString().find("/some/dir"), std::string::npos);
  }

  // NotADir
  {
    auto status = arrow::fs::internal::NotADir("/some/file");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.ToString().find("/some/file"), std::string::npos);
  }

  // NotEmpty
  {
    auto status = arrow::fs::internal::NotEmpty("/some/dir");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.ToString().find("/some/dir"), std::string::npos);
  }

  // NotAFile
  {
    auto status = arrow::fs::internal::NotAFile("/some/dir");
    EXPECT_TRUE(status.IsIOError());
    EXPECT_NE(status.ToString().find("/some/dir"), std::string::npos);
  }
}

TEST_F(S3UnitTest, TestUriParsing) {
  // ParseFileSystemUri — valid
  {
    auto result = arrow::fs::internal::ParseFileSystemUri("s3://mybucket/path");
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->scheme(), "s3");
  }

  // ParseFileSystemUri — invalid
  {
    auto result = arrow::fs::internal::ParseFileSystemUri("://bad-uri");
    EXPECT_FALSE(result.ok());
  }

  // PathFromUriHelper — absolute path accepted
  {
    auto result = arrow::fs::internal::PathFromUriHelper("/some/local/path", {"file"}, true,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kDisallow);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, "/some/local/path");
  }

  // PathFromUriHelper — absolute path not accepted
  {
    auto result = arrow::fs::internal::PathFromUriHelper("/some/local/path", {"s3"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kDisallow);
    EXPECT_FALSE(result.ok());
  }

  // PathFromUriHelper — supported scheme
  {
    auto result = arrow::fs::internal::PathFromUriHelper("s3://mybucket/path", {"s3"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kPrepend);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, "mybucket/path");
  }

  // PathFromUriHelper — unsupported scheme
  {
    auto result = arrow::fs::internal::PathFromUriHelper("hdfs://namenode/path", {"s3", "file"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kPrepend);
    EXPECT_FALSE(result.ok());
  }

  // PathFromUriHelper — disallow authority
  {
    auto result = arrow::fs::internal::PathFromUriHelper("file://somehost/path", {"file"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kDisallow);
    EXPECT_FALSE(result.ok());
  }

  // PathFromUriHelper — ignore authority
  {
    auto result = arrow::fs::internal::PathFromUriHelper("s3://mybucket/path", {"s3"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kIgnore);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, "/path");
  }

  // PathFromUriHelper — windows authority
  {
    auto result = arrow::fs::internal::PathFromUriHelper("file://server/share/path", {"file"}, false,
                                                         arrow::fs::internal::AuthorityHandlingBehavior::kWindows);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(*result, "//server/share/path");
  }
}

TEST_F(S3UnitTest, TestCopyStream) {
  auto src_buf = arrow::Buffer::FromString("hello world test data");
  auto src = std::make_shared<arrow::io::BufferReader>(src_buf);

  ASSERT_AND_ASSIGN(auto dest, arrow::io::BufferOutputStream::Create(1024));
  ASSERT_STATUS_OK(arrow::fs::internal::CopyStream(src, dest, 8, arrow::io::default_io_context()));

  ASSERT_AND_ASSIGN(auto result_buf, dest->Finish());
  EXPECT_EQ(result_buf->ToString(), "hello world test data");
}

TEST_F(S3UnitTest, TestCreateS3Options) {
  // No SSL → scheme=http
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAWS;
    config.access_key_id = "test_ak";
    config.access_key_value = "test_sk";
    config.request_timeout_ms = 5000;
    config.region = "us-east-1";

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->scheme, "http");
    EXPECT_EQ(result->endpoint_override, config.address);
    EXPECT_EQ(result->region, "us-east-1");
  }

  // Aliyun → force_virtual_addressing
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAliyun;
    config.access_key_id = "ak";
    config.access_key_value = "sk";
    config.region = "cn-hangzhou";

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->force_virtual_addressing);
  }

  // Tencent → force_virtual_addressing
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderTencent;
    config.access_key_id = "ak";
    config.access_key_value = "sk";
    config.region = "ap-guangzhou";

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->force_virtual_addressing);
  }

  // Huawei → force_virtual_addressing
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderHuawei;
    config.access_key_id = "ak";
    config.access_key_value = "sk";
    config.region = "cn-north-1";

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->force_virtual_addressing);
  }

  // Timeout 0 → use default
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAWS;
    config.access_key_id = "ak";
    config.access_key_value = "sk";
    config.request_timeout_ms = 0;

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_GT(result->request_timeout, 0);
  }

  // Explicit credentials (use_iam=false)
  {
    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAWS;
    config.access_key_id = "mykey";
    config.access_key_value = "mysecret";
    config.use_iam = false;

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetAccessKey(), "mykey");
    EXPECT_EQ(result->GetSecretKey(), "mysecret");
  }
}

TEST_F(S3UnitTest, TestS3GlobalOptions) {
  auto options = S3GlobalOptions::Defaults();
  (void)options;  // Just verify no crash
}

TEST_F(S3UnitTest, TestClientBuilder) {
  // Non-AWS S3-compatible backends must not probe AWS EC2 IMDS while
  // constructing their AWS SDK client configuration.
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderAWS;
    ClientBuilder builder(options);
    EXPECT_FALSE(builder.config().disableIMDS);
  }
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderAliyun;
    ClientBuilder builder(options);
    EXPECT_TRUE(builder.config().disableIMDS);
  }
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderGCP;
    ClientBuilder builder(options);
    EXPECT_TRUE(builder.config().disableIMDS);
  }
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderTencent;
    ClientBuilder builder(options);
    EXPECT_TRUE(builder.config().disableIMDS);
  }
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.cloud_provider = kCloudProviderHuawei;
    ClientBuilder builder(options);
    EXPECT_TRUE(builder.config().disableIMDS);
  }

  // Construct and access options/config
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    ClientBuilder<S3Client> builder(options);
    EXPECT_EQ(builder.options().GetAccessKey(), "ak");

    const auto& config = builder.config();
    (void)config.region;
    EXPECT_NE(builder.mutable_config(), nullptr);
  }

  // Build with region + endpoint
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.region = "us-west-2";
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";

    ClientBuilder<S3Client> builder(options);
    auto result = builder.BuildClient();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_NE(*result, nullptr);
  }

  // Build with https
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "https";
    options.endpoint_override = "s3.amazonaws.com";

    ClientBuilder<S3Client> builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with invalid scheme
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "ftp";

    ClientBuilder<S3Client> builder(options);
    EXPECT_FALSE(builder.BuildClient().ok());
  }

  // Build with proxy
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";
    options.proxy_options.scheme = "http";
    options.proxy_options.host = "proxy.example.com";
    options.proxy_options.port = 8080;
    options.proxy_options.username = "proxyuser";
    options.proxy_options.password = "proxypass";

    ClientBuilder<S3Client> builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with null credentials
  {
    auto options = S3Options::Defaults();
    options.scheme = "http";
    options.credentials_provider = nullptr;

    ClientBuilder<S3Client> builder(options);
    EXPECT_FALSE(builder.BuildClient().ok());
  }

  // Build with retry strategy
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(3);

    ClientBuilder<S3Client> builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with invalid proxy scheme
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";
    options.proxy_options.scheme = "ftp";

    ClientBuilder<S3Client> builder(options);
    EXPECT_FALSE(builder.BuildClient().ok());
  }
}

TEST_F(S3UnitTest, TestS3ClientHolder) {
  auto options = S3Options::FromAccessKey("ak", "sk");
  options.scheme = "http";
  options.endpoint_override = "localhost:9000";

  ClientBuilder<S3Client> builder(options);
  ASSERT_AND_ASSIGN(auto holder, builder.BuildClient());
  ASSERT_NE(holder, nullptr);

  // Lock
  {
    auto lock_result = holder->Lock();
    ASSERT_TRUE(lock_result.ok()) << lock_result.status().ToString();
    EXPECT_NE(lock_result->get(), nullptr);
  }

  // Lock + Move
  {
    ASSERT_AND_ASSIGN(auto lock, holder->Lock());
    auto* ptr_before = lock.get();
    auto moved = lock.Move();
    EXPECT_EQ(moved.get(), ptr_before);
  }
}

}  // namespace milvus_storage
