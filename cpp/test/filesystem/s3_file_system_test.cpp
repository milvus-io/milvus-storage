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

#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>

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
#include "milvus-storage/filesystem/s3/s3_client_builder.h"
#include "milvus-storage/filesystem/s3/s3_auth_signer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/util_internal.h"
#include "milvus-storage/filesystem/fs.h"

#include "test_env.h"

namespace milvus_storage {

namespace {

class ScopedEnvironmentVariable {
  public:
  ScopedEnvironmentVariable(std::string name, std::string value) : name_(std::move(name)) {
    if (const char* old = std::getenv(name_.c_str()); old != nullptr) {
      old_value_ = old;
    }
    setenv(name_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvironmentVariable() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

  private:
  std::string name_;
  std::optional<std::string> old_value_;
};

}  // namespace

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
    // Keep filtered runs of this suite safe too. Finalize once at process exit,
    // after every S3-using suite has finished and before AwsInstance's static
    // destructor.
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit([] { EnsureS3Finalized().ok(); }); });
  }
};

constexpr fs::internal::S3ErrorProvenance kDefaultProvenance{};

TEST_F(S3UnitTest, TestExtendErrorInFs) {
  Aws::Client::AWSError<Aws::S3::S3Errors> test_err(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                    Aws::Client::RetryableType::NOT_RETRYABLE, "StorageNoSuchUpload",
                                                    "Just for test");

  auto status = fs::internal::ErrorToStatus("test", test_err, kDefaultProvenance);
  ASSERT_STATUS_NOT_OK(status);
  auto extend_status = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(extend_status, nullptr);
  ASSERT_EQ(extend_status->code(), ExtendStatusCode::StorageNoSuchUpload);
  ASSERT_TRUE(status.ToString().find(extend_status->ToString()) != std::string::npos);
}

TEST_F(S3UnitTest, ExpiredTokenIsAuthenticationFailure) {
  auto expired_token = [](Aws::S3::S3Errors error_type) {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(error_type, Aws::Client::RetryableType::NOT_RETRYABLE,
                                                   "ExpiredToken", "token has expired");
    error.SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
    return error;
  };

  // Both shapes occur in practice: some SDK/parser combinations keep the
  // service error UNKNOWN, while others normalize the same named condition to
  // ACCESS_DENIED. Neither shape proves that the configured provider will
  // return a different credential on an immediate replay.
  for (auto error_type : {Aws::S3::S3Errors::UNKNOWN, Aws::S3::S3Errors::ACCESS_DENIED}) {
    auto error = expired_token(error_type);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, fs::internal::S3ErrorProvenance{});
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
    EXPECT_FALSE(detail->retryable());
  }

  // Some SDK versions expose the same named service condition through a core
  // credential enum. The exception name still carries the authoritative
  // ExpiredToken spelling and must still map to the authentication verdict.
  const Aws::Client::CoreErrors core_shapes[] = {
      Aws::Client::CoreErrors::INVALID_CLIENT_TOKEN_ID,
      Aws::Client::CoreErrors::MISSING_AUTHENTICATION_TOKEN,
  };
  for (auto core : core_shapes) {
    auto error = expired_token(static_cast<Aws::S3::S3Errors>(core));
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, fs::internal::S3ErrorProvenance{});
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
    EXPECT_FALSE(detail->retryable());
  }
}

TEST_F(S3UnitTest, WrongBucketRegionIsConfigurationFailure) {
  Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE,
                                                 "AuthorizationHeaderMalformed",
                                                 "the authorization header is malformed");
  error.SetResponseCode(Aws::Http::HttpResponseCode::BAD_REQUEST);
  Aws::Http::HeaderValueCollection headers;
  headers["x-amz-bucket-region"] = "us-west-2";
  error.SetResponseHeaders(headers);

  auto status = fs::internal::ErrorToStatus("When reading bucket 'b': ", "HeadBucket", error,
                                            fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::Bucket},
                                            std::string("us-east-1"));
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid);
  EXPECT_FALSE(detail->retryable());
  EXPECT_NE(status.message().find("configured region is 'us-east-1'"), std::string::npos) << status.ToString();
  EXPECT_NE(status.message().find("bucket is located in 'us-west-2'"), std::string::npos) << status.ToString();
  EXPECT_NE(detail->extra_info().find("configured_region=us-east-1"), std::string::npos);
  EXPECT_NE(detail->extra_info().find("actual_region=us-west-2"), std::string::npos);
}

// A store may echo x-amz-bucket-region on any response, including one that
// identifies a different, sharper condition. The region mismatch is judged last
// so it cannot erase that condition: a request that was throttled while pointed
// at the wrong region was still throttled, and reporting it as a configuration
// failure would strip the transient hint from a failure that has one.
TEST_F(S3UnitTest, WrongBucketRegionDoesNotMaskAnIdentifiedCondition) {
  Aws::Http::HeaderValueCollection headers;
  headers["x-amz-bucket-region"] = "us-west-2";

  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::SLOW_DOWN, Aws::Client::RetryableType::RETRYABLE,
                                                   "SlowDown", "please reduce your request rate");
    error.SetResponseCode(Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS);
    error.SetResponseHeaders(headers);

    auto status = fs::internal::ErrorToStatus("When reading key 'k' in bucket 'b': ", "GetObject", error,
                                              fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::Object},
                                              std::string("us-east-1"));
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
    EXPECT_TRUE(detail->retryable());
    // The mismatch is not lost, it is just not the verdict.
    EXPECT_NE(status.message().find("bucket is located in 'us-west-2'"), std::string::npos) << status.ToString();
    EXPECT_NE(detail->extra_info().find("actual_region=us-west-2"), std::string::npos);
  }

  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "access denied");
    error.SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
    error.SetResponseHeaders(headers);

    auto status = fs::internal::ErrorToStatus("When reading key 'k' in bucket 'b': ", "GetObject", error,
                                              fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::Object},
                                              std::string("us-east-1"));
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
    EXPECT_FALSE(detail->retryable());
  }
}

// A classified status names what it was operating on, and extra_info carries
// something the message does not.
//
// Both used to be missing: the prefix -- which is where the bucket and key are
// -- was glued onto the fallback IOError only, so every status that DID carry a
// verdict said "AccessDenied during HeadObject" and nothing about which object,
// while extra_info was the message a second time.
TEST_F(S3UnitTest, ClassifiedErrorsKeepPrefixAndExceptionName) {
  Aws::Client::AWSError<Aws::S3::S3Errors> error(
      Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
  auto status = fs::internal::ErrorToStatus("When reading information for key 'k' in bucket 'b': ", "HeadObject", error,
                                            kDefaultProvenance);
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();

  EXPECT_NE(status.message().find("key 'k'"), std::string::npos) << status.message();
  EXPECT_NE(status.message().find("bucket 'b'"), std::string::npos) << status.message();

  EXPECT_NE(detail->extra_info().find("AccessDenied"), std::string::npos) << detail->extra_info();
  EXPECT_NE(detail->extra_info().find("HeadObject"), std::string::npos) << detail->extra_info();
  EXPECT_NE(detail->extra_info(), status.message());

  // An endpoint that answers with a whole page does not get to put it all in a
  // Status that is then logged, wrapped and carried across the FFI boundary.
  const std::string huge(4096, 'x');
  Aws::Client::AWSError<Aws::S3::S3Errors> chatty(
      Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE,
      Aws::String(huge.begin(), huge.end()), Aws::String(huge.begin(), huge.end()));
  auto bounded = fs::internal::ErrorToStatus("prefix: ", "HeadObject", chatty, kDefaultProvenance);
  EXPECT_LT(bounded.message().size(), huge.size());
  auto bounded_detail = ExtendStatusDetail::UnwrapStatus(bounded);
  ASSERT_NE(bounded_detail, nullptr) << bounded.ToString();
  EXPECT_LT(bounded_detail->extra_info().size(), huge.size());

  // The caller-supplied prefix carries bucket/key context, but it is not
  // trusted input either. Bounding only the AWS response still allowed a huge
  // key or endpoint diagnostic to cross the FFI boundary and flood logs.
  auto bounded_prefix = fs::internal::ErrorToStatus(huge, "HeadObject", error, kDefaultProvenance);
  EXPECT_LT(bounded_prefix.message().size(), huge.size());
  EXPECT_NE(bounded_prefix.message().find("HeadObject"), std::string::npos) << bounded_prefix.message();
}

TEST_F(S3UnitTest, TestErrorToStatusNonRetryableVsRetryable) {
  // NoSuchKey: non-retryable, tagged StorageNotFound.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::NO_SUCH_KEY, Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchKey", "object gone");
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageNotFound);
  }
  // AccessDenied: System/non-retryable, tagged StorageAccessDenied. Operator
  // credentials still land on segcore's 2006 ConfigInvalid rather than 2044.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
  }
  // An otherwise unclassified error remains a bare IOError regardless of the
  // AWS SDK retry-policy flag. ShouldRetry is not an observed error cause.
  for (auto retry_policy : {Aws::Client::RetryableType::NOT_RETRYABLE, Aws::Client::RetryableType::RETRYABLE}) {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::VALIDATION, retry_policy, "ValidationError",
                                                   "bad request");
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    EXPECT_TRUE(status.IsIOError()) << status.ToString();
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    EXPECT_EQ(detail, nullptr) << status.ToString();
  }
  // MinIO-style SlowDown: arrives as UNKNOWN + non-retryable, but it is a
  // genuine transient (rate limiting), so it must carry retryable throttling
  // detail instead of being treated as a System failure.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "rate limited");
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
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
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
    EXPECT_TRUE(detail->retryable());
  }
}

// The SDK's own core errors, which are not S3 errors at all.
//
// CoreErrors and S3Errors share a numeric space only because the S3 enum
// continues where the core one stops, so casting a core code into it names
// whichever S3 error happens to sit at that number. These three then missed
// every arm and fell through to the unclassified IOError path -- a local
// allocation failure reported as generic storage I/O, and rejected credentials
// reported as a generic storage failure instead of something an operator can
// fix.
//
// This test exists because the first attempt at the fix gated on
// CoreErrors::VALIDATION (14) while the codes it handles are 17, 21 and 26 --
// the branch could not execute, and nothing said so.
TEST_F(S3UnitTest, CoreErrorsAreClassifiedBeforeTheS3Cast) {
  {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(
        Aws::Client::CoreErrors::MEMORY_ALLOCATION, Aws::Client::RetryableType::NOT_RETRYABLE, "OOM", "alloc failed");
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::MemAllocateFailed);
  }

  for (auto core : {Aws::Client::CoreErrors::UNRECOGNIZED_CLIENT, Aws::Client::CoreErrors::INVALID_SIGNATURE}) {
    Aws::Client::AWSError<Aws::Client::CoreErrors> error(core, Aws::Client::RetryableType::NOT_RETRYABLE, "Auth",
                                                         "bad credentials");
    error.SetResponseCode(Aws::Http::HttpResponseCode::FORBIDDEN);
    auto status =
        fs::internal::ErrorToStatus("When reading key 'k' in bucket 'b': ", "HeadObject", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
    EXPECT_FALSE(detail->retryable());
    EXPECT_NE(status.message().find("key 'k'"), std::string::npos) << status.message();
    EXPECT_NE(status.message().find("bucket 'b'"), std::string::npos) << status.message();
    EXPECT_NE(detail->extra_info().find("operation=HeadObject"), std::string::npos) << detail->extra_info();
    EXPECT_NE(detail->extra_info().find("exception=Auth"), std::string::npos) << detail->extra_info();
    EXPECT_NE(detail->extra_info().find("http_status=403"), std::string::npos) << detail->extra_info();
    EXPECT_NE(detail->extra_info(), status.message());
  }
}

TEST_F(S3UnitTest, Conflict409IsNotForNonConflictBucketStates) {
  // A 409 with no recognized non-conflict name is a race. It remains Conflict so
  // a business-aware caller can coordinate, but generic retry stays disabled.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "OperationAborted",
                                                   "conditional request conflict");
    error.SetResponseCode(Aws::Http::HttpResponseCode::CONFLICT);
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConflict);
    EXPECT_FALSE(detail->retryable());
  }

  // But not every 409 is a race. BucketNotEmpty and InvalidBucketState are
  // answers about the bucket -- replaying the same request cannot change
  // either, so classifying them Conflict would send a non-conflict condition to
  // business coordination.
  for (const char* name : {"BucketNotEmpty", "InvalidBucketState"}) {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, name, "non-conflict 409");
    error.SetResponseCode(Aws::Http::HttpResponseCode::CONFLICT);
    auto status = fs::internal::ErrorToStatus("test", error, kDefaultProvenance);
    SCOPED_TRACE(name);
    ASSERT_STATUS_NOT_OK(status);
    // Today these land unclassified (plain IOError): the blocklist returns
    // nullopt and the SDK-verdict fallback is gated on a recognized error
    // type. Pinned so a future reclassification is a conscious choice -- the
    // one outcome that must never come back is retryable Conflict.
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    EXPECT_EQ(detail, nullptr);
    if (detail != nullptr) {
      EXPECT_NE(detail->code(), ExtendStatusCode::StorageConflict);
      EXPECT_FALSE(detail->retryable());
    }
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

TEST_F(S3UnitTest, TestS3ErrorClassification) {
  // Bucket and object absence are separate control-flow predicates. A missing
  // bucket must never be swallowed by an object-level allow_not_found gate.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::NO_SUCH_BUCKET, Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchBucket", "not found");
    EXPECT_TRUE(fs::internal::IsBucketNotFound(error));
    EXPECT_TRUE(fs::internal::IsExplicitBucketNotFound(error));
    EXPECT_FALSE(fs::internal::IsObjectNotFound(error));
  }
  // Generic RESOURCE_NOT_FOUND is ambiguous until the operation supplies the
  // resource kind.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::RESOURCE_NOT_FOUND,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "ResourceNotFound",
                                                   "not found");
    EXPECT_TRUE(fs::internal::IsBucketNotFound(error));
    EXPECT_TRUE(fs::internal::IsObjectNotFound(error));
    EXPECT_FALSE(fs::internal::IsExplicitBucketNotFound(error));
  }
  // NO_SUCH_KEY was previously omitted from the existence predicate.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NO_SUCH_KEY,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchKey", "not found");
    EXPECT_TRUE(fs::internal::IsObjectNotFound(error));
    EXPECT_FALSE(fs::internal::IsBucketNotFound(error));
  }
  // Neither kind of not-found.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    EXPECT_FALSE(fs::internal::IsBucketNotFound(error));
    EXPECT_FALSE(fs::internal::IsObjectNotFound(error));
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

  // The same generic 404 means opposite actions depending on the operation:
  // HeadObject -> object-not-found handling, HeadBucket/ListObjects -> fix the
  // deployment bucket (BucketNotFound).
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::RESOURCE_NOT_FOUND,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "ResourceNotFound",
                                                   "not found");
    auto object_status = fs::internal::ErrorToStatus(
        "object: ", "HeadObject", error, fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::Object});
    auto object_detail = ExtendStatusDetail::UnwrapStatus(object_status);
    ASSERT_NE(object_detail, nullptr);
    EXPECT_EQ(object_detail->code(), ExtendStatusCode::StorageNotFound);

    auto bucket_status = fs::internal::ErrorToStatus(
        "bucket: ", "HeadBucket", error, fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::Bucket});
    auto bucket_detail = ExtendStatusDetail::UnwrapStatus(bucket_status);
    ASSERT_NE(bucket_detail, nullptr);
    EXPECT_EQ(bucket_detail->code(), ExtendStatusCode::StorageBucketNotFound);

    auto upload_status =
        fs::internal::ErrorToStatus("upload: ", "UploadPart", error,
                                    fs::internal::S3ErrorProvenance{fs::internal::S3ResourceKind::MultipartUpload});
    auto upload_detail = ExtendStatusDetail::UnwrapStatus(upload_status);
    ASSERT_NE(upload_detail, nullptr);
    EXPECT_EQ(upload_detail->code(), ExtendStatusCode::StorageNoSuchUpload);
  }

  // NO_SUCH_UPLOAD -- classified, but NOT through AssertRetryableCode. The
  // upload id the caller held is gone; resending against it fails identically
  // every time, so only starting a fresh upload helps and that decision belongs
  // to the layer that owns the write. System, not retriable.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NO_SUCH_UPLOAD,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchUpload",
                                                   "Upload not found");
    auto status = fs::internal::ErrorToStatus("test_prefix", "CompleteMultipart", error, kDefaultProvenance);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageNoSuchUpload);
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
    EXPECT_FALSE(detail->retryable());
  }

  // NO_SUCH_BUCKET is deliberately NOT the same code as a missing key: nothing
  // was lost, and re-reading metadata cannot conjure a bucket. It is a
  // deployment pointing somewhere that does not exist.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::NO_SUCH_BUCKET, Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchBucket", "bucket gone");
    auto status = fs::internal::ErrorToStatus("test_prefix", "GetObject", error, kDefaultProvenance);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageBucketNotFound);
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
    EXPECT_FALSE(detail->retryable());
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::BucketInvalid);  // 2016, not 2017
  }

  // A missing KEY keeps ObjectNotExist -- the two must not collapse back.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NO_SUCH_KEY,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NoSuchKey", "key gone");
    auto status = fs::internal::ErrorToStatus("test_prefix", "GetObject", error, kDefaultProvenance);
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageNotFound);
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist);  // 2017
  }

  // AWS SDK retryable error
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::INTERNAL_FAILURE, Aws::Client::RetryableType::RETRYABLE, "InternalFailure", "retryable");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // ShouldRetry alone is policy, not evidence that the underlying condition
  // is a network or otherwise transient failure.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::RETRYABLE,
                                                   "UnclassifiedRetryable", "SDK policy allows retry");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    ASSERT_STATUS_NOT_OK(status);
    EXPECT_TRUE(status.IsIOError());
    EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  }

  // NETWORK_CONNECTION
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::NETWORK_CONNECTION,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "NetworkConnection",
                                                   "network");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientNetwork);
  }

  // REQUEST_TIMEOUT
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::REQUEST_TIMEOUT, Aws::Client::RetryableType::NOT_RETRYABLE, "RequestTimeout", "timeout");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientTimeout);
  }

  // HTTP 408
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "RequestTimeout", "timeout");
    error.SetResponseCode(Aws::Http::HttpResponseCode::REQUEST_TIMEOUT);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientTimeout);
  }

  // HTTP 429 from S3-compatible backends may arrive as UNKNOWN.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "TooManyRequests", "rate limited");
    error.SetResponseCode(Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // SLOW_DOWN
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::SLOW_DOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // THROTTLING
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::THROTTLING, Aws::Client::RetryableType::NOT_RETRYABLE, "Throttling", "throttled");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // MinIO SlowDown
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDown", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // MinIO SlowDownWrite
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "SlowDownWrite", "slow");
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientThrottling);
  }

  // SERVICE_UNAVAILABLE
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::SERVICE_UNAVAILABLE,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "ServiceUnavailable",
                                                   "unavailable");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 500
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "InternalError", "internal");
    error.SetResponseCode(Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 502
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "BadGateway", "bad gateway");
    error.SetResponseCode(Aws::Http::HttpResponseCode::BAD_GATEWAY);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 503
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "ServiceUnavailable", "unavailable");
    error.SetResponseCode(Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // HTTP 504
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "GatewayTimeout", "gateway timeout");
    error.SetResponseCode(Aws::Http::HttpResponseCode::GATEWAY_TIMEOUT);
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // XMinioServerNotInitialized
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE,
                                                   "XMinioServerNotInitialized", "server not initialized");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
    AssertRetryableCode(status, ExtendStatusCode::StorageTransientService);
  }

  // PRECONDITION_FAILED
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "PreconditionFailed",
                                                   "condition failed");
    error.SetResponseCode(Aws::Http::HttpResponseCode::PRECONDITION_FAILED);
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StoragePreConditionFailed);
    EXPECT_FALSE(detail->retryable());
  }

  // CONFLICT
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(Aws::S3::S3Errors::UNKNOWN,
                                                   Aws::Client::RetryableType::NOT_RETRYABLE, "Conflict", "conflict");
    error.SetResponseCode(Aws::Http::HttpResponseCode::CONFLICT);
    auto status = fs::internal::ErrorToStatus("prefix", "PutObject", error, kDefaultProvenance);
    ASSERT_FALSE(status.ok());
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConflict);
    EXPECT_FALSE(detail->retryable());
  }

  // Generic UNKNOWN IOError with no recognized permanent or transient signal
  // remains plain IOError with no ExtendStatus detail.
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::UNKNOWN, Aws::Client::RetryableType::NOT_RETRYABLE, "SomeBackendSpecificError", "opaque");
    auto status = fs::internal::ErrorToStatus("prefix", "GetObject", error, kDefaultProvenance);
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
    EXPECT_TRUE(fs::internal::OutcomeToStatus("prefix", "PutObject", outcome, kDefaultProvenance).ok());
  }

  // Failure
  {
    Aws::Client::AWSError<Aws::S3::S3Errors> error(
        Aws::S3::S3Errors::ACCESS_DENIED, Aws::Client::RetryableType::NOT_RETRYABLE, "AccessDenied", "forbidden");
    Aws::Utils::Outcome<Aws::S3::Model::PutObjectResult, Aws::Client::AWSError<Aws::S3::S3Errors>> outcome(
        std::move(error));
    EXPECT_FALSE(fs::internal::OutcomeToStatus("prefix", "PutObject", outcome, kDefaultProvenance).ok());
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
  auto assert_config_error = [](const arrow::Status& status) {
    ASSERT_FALSE(status.ok()) << status.ToString();
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid) << status.ToString();
    EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
    EXPECT_FALSE(detail->retryable());
  };

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

  // Producer-side deployment mistakes must not fall through as an untyped
  // untyped Invalid/System error.
  {
    ScopedEnvironmentVariable auth_mode("ALIYUN_ROLE_ARN_AUTH_MODE", "");
    ScopedEnvironmentVariable token_file("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "");
    ScopedEnvironmentVariable provider_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "");
    ScopedEnvironmentVariable machine_role("ALIBABA_CLOUD_ROLE_ARN", "");

    ArrowFileSystemConfig config;
    config.cloud_provider = kCloudProviderAliyun;
    config.role_arn = "acs:ram::123456:role/target";
    S3FileSystemProducer producer(config);
    assert_config_error(producer.CreateS3Options().status());
  }
  {
    ArrowFileSystemConfig config;
    config.cloud_provider = "unsupported-cloud";
    config.role_arn = "role";
    S3FileSystemProducer producer(config);
    assert_config_error(producer.CreateS3Options().status());
  }
  {
    ArrowFileSystemConfig config;
    config.cloud_provider = "unsupported-cloud";
    config.use_iam = true;
    S3FileSystemProducer producer(config);
    assert_config_error(producer.CreateS3Options().status());
  }

  // AWS use_iam still selects the environment provider first when static
  // credentials are present.
  {
    ScopedEnvironmentVariable access_key("AWS_ACCESS_KEY_ID", "environment-ak");
    ScopedEnvironmentVariable secret_key("AWS_SECRET_ACCESS_KEY", "environment-sk");
    ScopedEnvironmentVariable session_token("AWS_SESSION_TOKEN", "expired-static-token");

    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAWS;
    config.use_iam = true;

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->credentials_kind, S3CredentialsKind::Default);
  }

  // Long-lived static credentials are a complete AWS credential without a
  // session token. The producer must accept this ordinary default-chain result.
  {
    ScopedEnvironmentVariable access_key("AWS_ACCESS_KEY_ID", "environment-long-lived-ak");
    ScopedEnvironmentVariable secret_key("AWS_SECRET_ACCESS_KEY", "environment-long-lived-sk");
    ScopedEnvironmentVariable session_token("AWS_SESSION_TOKEN", "");

    ArrowFileSystemConfig config;
    config.use_ssl = false;
    config.cloud_provider = kCloudProviderAWS;
    config.use_iam = true;

    S3FileSystemProducer producer(config);
    auto result = producer.CreateS3Options();
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->GetAccessKey(), "environment-long-lived-ak");
    EXPECT_EQ(result->GetSecretKey(), "environment-long-lived-sk");
    EXPECT_TRUE(result->GetSessionToken().empty());
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

    auto fs_result = S3FileSystem::Make(options);
    ASSERT_FALSE(fs_result.ok());
    EXPECT_TRUE(fs_result.status().IsInvalid()) << fs_result.status().ToString();
    EXPECT_NE(fs_result.status().message().find("Failed to build S3 client"), std::string::npos)
        << fs_result.status().ToString();
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
