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

#include <memory>
#include <sstream>
#include <type_traits>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/s3/model/PutObjectResult.h>

#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/filesystem/upload_sizable.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/s3_options.h"
#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/util_internal.h"
#include "milvus-storage/filesystem/fs.h"

#include "test_env.h"

namespace milvus_storage {

// ============================================================================
// Cloud-env tests (existing)
// ============================================================================

class S3FsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsCloudEnv()) {
      GTEST_SKIP() << "S3 tests skipped in non-cloud environment";
    }
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));
  }

  milvus_storage::api::Properties properties_;
  ArrowFileSystemPtr fs_;
};

TEST_F(S3FsTest, ConditionalWrite) {
  std::string file_to = "/test_conditional_write.txt";

  // Ensure source file does not exist
  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  // Create source file
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
    // check file exists, it should be a file
    ASSERT_AND_ASSIGN(auto file_info, fs_->GetFileInfo(file_to));
    ASSERT_EQ(file_info.type(), arrow::fs::FileType::File);
  }

  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_NOT_OK(output_stream->Close());
  }

  (void)fs_->DeleteFile(file_to);

  // Test conditional write in output_stream close
  {
    std::shared_ptr<arrow::Buffer> buffer1 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());
    std::shared_ptr<arrow::Buffer> buffer2 =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    auto conditional_fs = std::dynamic_pointer_cast<UploadConditional>(fs_);
    ASSERT_NE(conditional_fs, nullptr);
    ASSERT_AND_ASSIGN(auto output_stream1, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream1->Write(buffer1));

    ASSERT_AND_ASSIGN(auto output_stream2, conditional_fs->OpenConditionalOutputStream(file_to, nullptr));
    ASSERT_STATUS_OK(output_stream2->Write(buffer2));

    ASSERT_STATUS_OK(output_stream1->Close());
    auto write_status = output_stream2->Close();
    ASSERT_FALSE(write_status.ok());
    auto extend_status = ExtendStatusDetail::UnwrapStatus(write_status);
    ASSERT_NE(extend_status, nullptr);
    ASSERT_TRUE(extend_status->code() == ExtendStatusCode::AwsErrorPreConditionFailed ||
                extend_status->code() == ExtendStatusCode::AwsErrorConflict);
  }
}

TEST_F(S3FsTest, TestMetadata) {
  // predefined metadata
  {
    std::string file_to = "/predefined_metadata.txt";
    (void)fs_->DeleteFile(file_to);
    std::string content = "This is a test file for metadata.";

    auto kvmeta = arrow::KeyValueMetadata::Make({"Content-Language"}, {"zh-CN"});
    ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_to, kvmeta));
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.c_str()), content.size());

    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }

  // custom metadata
  {
    std::string file_to = "/custom_metadata.txt";
    (void)fs_->DeleteFile(file_to);
    std::string content = "This is a test file for custom metadata.";
    auto kvmeta = arrow::KeyValueMetadata::Make({"Content-Disposition"}, {"inline"});
    ASSERT_AND_ASSIGN(auto output_stream, fs_->OpenOutputStream(file_to, kvmeta));
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content.c_str()), content.size());
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }
}

TEST_F(S3FsTest, TestExtendErrorInFs) {
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

// ============================================================================
// Non-cloud unit tests — S3 SDK initialized but no real cloud connection needed
// ============================================================================

class S3UnitTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() { ASSERT_TRUE(EnsureS3Initialized().ok()); }
};

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
  // NO_SUCH_UPLOAD → ExtendStatus
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchUpload",
                                                   "Upload not found");
    auto status = fs::internal::ErrorToStatus("test_prefix", "CompleteMultipart", error);
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::AwsErrorNoSuchUpload);
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
  }

  // Generic IOError (no ExtendStatus)
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error);
    ASSERT_FALSE(status.ok());
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
  // Construct and access options/config
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    ClientBuilder builder(options);
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

    ClientBuilder builder(options);
    auto result = builder.BuildClient();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_NE(*result, nullptr);
  }

  // Build with https
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "https";
    options.endpoint_override = "s3.amazonaws.com";

    ClientBuilder builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with invalid scheme
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "ftp";

    ClientBuilder builder(options);
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

    ClientBuilder builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with null credentials
  {
    auto options = S3Options::Defaults();
    options.scheme = "http";
    options.credentials_provider = nullptr;

    ClientBuilder builder(options);
    EXPECT_FALSE(builder.BuildClient().ok());
  }

  // Build with retry strategy
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(3);

    ClientBuilder builder(options);
    EXPECT_TRUE(builder.BuildClient().ok());
  }

  // Build with invalid proxy scheme
  {
    auto options = S3Options::FromAccessKey("ak", "sk");
    options.scheme = "http";
    options.endpoint_override = "localhost:9000";
    options.proxy_options.scheme = "ftp";

    ClientBuilder builder(options);
    EXPECT_FALSE(builder.BuildClient().ok());
  }
}

TEST_F(S3UnitTest, TestS3ClientHolder) {
  auto options = S3Options::FromAccessKey("ak", "sk");
  options.scheme = "http";
  options.endpoint_override = "localhost:9000";

  ClientBuilder builder(options);
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
