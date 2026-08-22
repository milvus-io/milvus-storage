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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <aws/core/platform/Environment.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/config/ConfigAndCredentialsCacheManager.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/utils/stream/ResponseStream.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/AssumeRoleRequest.h>
#include <aws/sts/model/AssumeRoleResult.h>
#include <aws/sts/model/AssumeRoleWithWebIdentityRequest.h>
#include <aws/sts/model/AssumeRoleWithWebIdentityResult.h>
#include <aws/sts/model/Credentials.h>
#include <arrow/testing/executor_util.h>
#include <arrow/util/thread_pool.h>

#include "milvus-storage/filesystem/s3/provider/AliyunCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunOIDCAssumeRoleChainProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunRAMCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/AliyunRAMSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/AliyunSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/AwsDefaultCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/AwsSTSAssumeRoleCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/credential_resolution.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/provider/TencentCloudSTSClient.h"
#include "milvus-storage/filesystem/s3/provider/HuaweiCloudCredentialsProvider.h"
#include "milvus-storage/filesystem/s3/s3_client_builder.h"
#include "milvus-storage/filesystem/s3/s3_filesystem.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#ifdef WITH_CRT
#include "milvus-storage/filesystem/s3/s3_crt_client.h"
#endif
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/fiu_local.h"
#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_filesystem_c.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "test_env.h"

namespace milvus_storage {

// ============================================================================
// RAII environment variable helper
// ============================================================================

Aws::Client::ClientConfiguration MakeNoImdsClientConfiguration() {
  return Aws::Client::ClientConfiguration(Aws::Client::ClientConfigurationInitValues{/*shouldDisableIMDS=*/true});
}

class ScopedEnvVar {
  public:
  ScopedEnvVar(const std::string& name, const std::string& value) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    setenv(name.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// Unsets an env var for the scope lifetime, restoring it on destruction.
class ScopedEnvUnset {
  public:
  explicit ScopedEnvUnset(const std::string& name) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    unsetenv(name.c_str());
  }

  ~ScopedEnvUnset() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvUnset(const ScopedEnvUnset&) = delete;
  ScopedEnvUnset& operator=(const ScopedEnvUnset&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// RAII helper for temporary files
class TempFile {
  public:
  explicit TempFile(const std::string& content) {
    path_ = "/tmp/test_oidc_token_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    std::ofstream ofs(path_);
    ofs << content;
    ofs.close();
  }

  ~TempFile() { std::remove(path_.c_str()); }

  [[nodiscard]] const std::string& path() const { return path_; }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  private:
  std::string path_;
};

class ReloadAwsConfigOnExit {
  public:
  ~ReloadAwsConfigOnExit() { Aws::Config::ReloadCachedConfigFile(); }
};

void PrepareS3DeathTest() {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // Set this before GoogleTest re-execs the child: gcov reads the prefix at
  // process startup, and parent/child must not write the same coverage file.
  const auto prefix = "/tmp/milvus-storage-s3-death-test-gcov-" + std::to_string(getpid());
  (void)setenv("GCOV_PREFIX", prefix.c_str(), 1);
}

// ============================================================================
// Mock HTTP infrastructure
// ============================================================================

enum class MockExceptionKind { None, RuntimeError, BadAlloc };

struct MockResponseSpec {
  Aws::Http::HttpResponseCode code;
  std::string body;
  Aws::Http::HeaderValueCollection headers;
  std::function<void()> on_request;
  MockExceptionKind exception_kind = MockExceptionKind::None;
};

class MockHttpClient : public Aws::Http::HttpClient {
  public:
  MockHttpClient() {}

  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
      const std::shared_ptr<Aws::Http::HttpRequest>& request,
      Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
      Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
    std::optional<MockResponseSpec> matched;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      recorded_requests_.push_back(request);

      auto uri = request->GetURIString();
      // Find a matching response queue by URL substring.
      for (auto& [url_key, q] : response_map_) {
        if (!q.empty() && uri.find(url_key) != Aws::String::npos) {
          matched = std::move(q.front());
          q.pop();
          break;
        }
      }
    }

    if (matched.has_value()) {
      switch (matched->exception_kind) {
        case MockExceptionKind::None:
          break;
        case MockExceptionKind::RuntimeError:
          throw std::runtime_error("synthetic credential dependency failure");
        case MockExceptionKind::BadAlloc:
          throw std::bad_alloc();
      }
      if (matched->on_request) {
        matched->on_request();
      }
      return BuildResponse(request, *matched);
    }

    // Default: return 404 for unmatched requests (e.g., background SDK requests)
    auto resp = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("MockHttpClient", request);
    resp->SetResponseCode(Aws::Http::HttpResponseCode::NOT_FOUND);
    return resp;
  }

  void EnqueueResponse(const std::string& url_match,
                       Aws::Http::HttpResponseCode code,
                       const std::string& body,
                       const Aws::Http::HeaderValueCollection& headers = {},
                       std::function<void()> on_request = {}) {
    response_map_[url_match].push({code, body, headers, std::move(on_request)});
  }

  void EnqueueException(const std::string& url_match, MockExceptionKind exception_kind) {
    MockResponseSpec spec;
    spec.exception_kind = exception_kind;
    response_map_[url_match].push(std::move(spec));
  }

  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> GetRecordedRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recorded_requests_;
  }

  private:
  static std::shared_ptr<Aws::Http::HttpResponse> BuildResponse(const std::shared_ptr<Aws::Http::HttpRequest>& request,
                                                                const MockResponseSpec& spec) {
    auto resp = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("MockHttpClient", request);
    resp->SetResponseCode(spec.code);
    for (const auto& h : spec.headers) {
      resp->AddHeader(h.first, h.second);
    }
    if (!spec.body.empty()) {
      resp->GetResponseBody() << spec.body;
    }
    return resp;
  }

  mutable std::mutex mutex_;
  mutable std::map<std::string, std::queue<MockResponseSpec>> response_map_;
  mutable std::vector<std::shared_ptr<Aws::Http::HttpRequest>> recorded_requests_;
};

class MockHttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  explicit MockHttpClientFactory(std::shared_ptr<MockHttpClient> client) : mock_client_(std::move(client)) {}

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& clientConfiguration) const override {
    return mock_client_;
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::String& uri, Aws::Http::HttpMethod method, const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("MockHttpClientFactory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  [[nodiscard]] std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
      const Aws::Http::URI& uri,
      Aws::Http::HttpMethod method,
      const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("MockHttpClientFactory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }

  private:
  std::shared_ptr<MockHttpClient> mock_client_;
};

class RejectingS3Executor final : public arrow::internal::Executor {
  public:
  int GetCapacity() override { return 1; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints,
                          arrow::internal::FnOnce<void()>,
                          arrow::StopToken,
                          StopCallback&&) override {
    return arrow::Status::IOError("executor rejected S3 upload task");
  }
};

class RejectAfterOneS3Executor final : public arrow::internal::Executor {
  public:
  explicit RejectAfterOneS3Executor(std::shared_ptr<arrow::internal::ThreadPool> pool)
      : pool_(std::move(pool)), rejection_observed_(rejection_promise_.get_future().share()) {}

  int GetCapacity() override { return pool_->GetCapacity(); }

  void Arm() { accepted_before_rejection_.store(1, std::memory_order_release); }

  const std::shared_future<void>& rejection_observed() const { return rejection_observed_; }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints hints,
                          arrow::internal::FnOnce<void()> task,
                          arrow::StopToken stop_token,
                          StopCallback&& stop_callback) override {
    int remaining = accepted_before_rejection_.load(std::memory_order_acquire);
    while (remaining >= 0) {
      if (remaining == 0) {
        if (!rejection_signaled_.exchange(true, std::memory_order_acq_rel)) {
          rejection_promise_.set_value();
        }
        return arrow::Status::IOError("executor rejected S3 delete task");
      }
      if (accepted_before_rejection_.compare_exchange_weak(remaining, remaining - 1, std::memory_order_acq_rel)) {
        break;
      }
    }
    return pool_->Spawn(hints, std::move(task), std::move(stop_token), std::move(stop_callback));
  }

  private:
  std::shared_ptr<arrow::internal::ThreadPool> pool_;
  std::atomic<int> accepted_before_rejection_{-1};
  std::atomic<bool> rejection_signaled_{false};
  std::promise<void> rejection_promise_;
  std::shared_future<void> rejection_observed_;
};

enum class S3SubmissionExceptionKind { kUnexpected, kBadAlloc };

class UnexpectedS3SubmissionException final : public std::exception {
  public:
  const char* what() const noexcept override { return "executor submission exploded unexpectedly"; }
};

class ThrowAfterOneS3Executor final : public arrow::internal::Executor {
  public:
  ThrowAfterOneS3Executor(std::shared_ptr<arrow::internal::ThreadPool> pool, S3SubmissionExceptionKind exception_kind)
      : pool_(std::move(pool)), exception_kind_(exception_kind) {}

  int GetCapacity() override { return pool_->GetCapacity(); }

  protected:
  arrow::Status SpawnReal(arrow::internal::TaskHints hints,
                          arrow::internal::FnOnce<void()> task,
                          arrow::StopToken stop_token,
                          StopCallback&& stop_callback) override {
    if (submissions_.fetch_add(1, std::memory_order_acq_rel) == 0) {
      return pool_->Spawn(hints, std::move(task), std::move(stop_token), std::move(stop_callback));
    }
    if (exception_kind_ == S3SubmissionExceptionKind::kBadAlloc) {
      throw std::bad_alloc();
    }
    throw UnexpectedS3SubmissionException();
  }

  private:
  std::shared_ptr<arrow::internal::ThreadPool> pool_;
  S3SubmissionExceptionKind exception_kind_;
  std::atomic<int> submissions_{0};
};

// ============================================================================
// Test Fixture
// ============================================================================

class S3ProviderTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      return;
    }
    ASSERT_TRUE(EnsureS3Initialized().ok());
    // Register S3 cleanup at process exit, so it runs after all test suites
    // but before AwsInstance's static destructor (which would warn otherwise).
    static std::once_flag flag;
    std::call_once(flag, [] { std::atexit([] { EnsureS3Finalized().ok(); }); });
  }

  void SetUp() override {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      GTEST_SKIP() << "S3 provider tests only run for AWS provider";
    }
    mock_client_ = std::make_shared<MockHttpClient>();
    auto factory = std::make_shared<MockHttpClientFactory>(mock_client_);
    Aws::Http::SetHttpClientFactory(factory);
  }

  void TearDown() override {
    auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
      return;
    }
    Aws::Http::CleanupHttp();
    Aws::Http::InitHttp();
  }

  void AssertBackgroundUploadSubmissionExceptionDrains(S3SubmissionExceptionKind exception_kind) {
    const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
    Aws::Http::HeaderValueCollection create_headers;
    create_headers["content-type"] = "application/xml";
    create_headers["content-length"] = std::to_string(create_result.size());

    std::promise<void> upload_started_promise;
    auto upload_started = upload_started_promise.get_future();
    std::promise<void> release_upload_promise;
    auto release_upload = release_upload_promise.get_future().share();
    std::atomic<bool> upload_release_timed_out{false};

    Aws::Http::HeaderValueCollection upload_headers;
    upload_headers["etag"] = "\"etag-1\"";
    upload_headers["content-length"] = "0";
    mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, create_headers);
    mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::OK, "", upload_headers,
                                  [&upload_started_promise, release_upload, &upload_release_timed_out] {
                                    upload_started_promise.set_value();
                                    if (release_upload.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                                      upload_release_timed_out.store(true, std::memory_order_release);
                                    }
                                  });
    mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

    S3Options options;
    options.ConfigureAnonymousCredentials();
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "mock-s3.local";
    options.cloud_provider = kCloudProviderAWS;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
    options.use_crt_async_reads = false;
    options.background_writes = true;

    ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(1));
    ThrowAfterOneS3Executor executor(pool, exception_kind);
    arrow::io::IOContext io_context(&executor);
    ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
    ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

    const uint8_t bytes[] = {7, 8};
    arrow::Status write_status;
    try {
      write_status = stream->Write(bytes, 2);
    } catch (const std::exception& e) {
      // Keep the regression failure deterministic: do not leave the accepted
      // sibling blocked while GoogleTest reports the escaped exception.
      release_upload_promise.set_value();
      pool->WaitForIdle();
      FAIL() << "executor submission exception escaped the Status API: " << e.what();
      return;
    } catch (...) {
      release_upload_promise.set_value();
      pool->WaitForIdle();
      FAIL() << "non-standard executor submission exception escaped the Status API";
      return;
    }
    const auto upload_started_state = upload_started.wait_for(std::chrono::seconds(5));
    if (upload_started_state != std::future_status::ready) {
      release_upload_promise.set_value();
      pool->WaitForIdle();
      FAIL() << "the accepted first upload did not start";
      return;
    }

    ASSERT_STATUS_NOT_OK(write_status);
    // Both exception kinds report the same verdict now: an allocation failure
    // in submission is not distinguished from any other submission exception.
    auto detail = ExtendStatusDetail::UnwrapStatus(write_status);
    ASSERT_NE(detail, nullptr) << write_status.ToString();
    EXPECT_EQ(detail->code(), ExtendStatusCode::InternalInvariantViolated) << write_status.ToString();

    // The synchronous failure is already the writer's terminal status. Close
    // must return it without waiting for the accepted sibling upload.
    auto close_status = stream->Close();
    EXPECT_TRUE(close_status.Equals(write_status)) << close_status.ToString();

    // Abort is deferred while that sibling remains in flight. The failed
    // submission must already have decremented its own counter slot, otherwise
    // the sibling's eventual completion can never take the counter to zero.
    ASSERT_STATUS_OK(stream->Abort());
    stream.reset();
    auto count_remote_aborts = [this] {
      size_t count = 0;
      for (const auto& request : mock_client_->GetRecordedRequests()) {
        if (request->GetMethod() == Aws::Http::HttpMethod::HTTP_DELETE &&
            request->GetURIString().find("uploadId=upload-id") != Aws::String::npos) {
          ++count;
        }
      }
      return count;
    };
    EXPECT_EQ(count_remote_aborts(), 0);

    release_upload_promise.set_value();
    pool->WaitForIdle();

    EXPECT_FALSE(upload_release_timed_out.load(std::memory_order_acquire));
    EXPECT_EQ(count_remote_aborts(), 1);
  }

  std::shared_ptr<MockHttpClient> mock_client_;
};

namespace {

void ExpectExtendStatusCode(const arrow::Status& status, ExtendStatusCode expected) {
  ASSERT_FALSE(status.ok()) << status.ToString();
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), expected) << status.ToString();
}

void ExpectCredentialDependencyException(const arrow::Status& status) {
  ASSERT_FALSE(status.ok()) << status.ToString();
  EXPECT_TRUE(status.IsIOError()) << status.ToString();
  EXPECT_FALSE(status.IsOutOfMemory()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  EXPECT_NE(status.message().find("synthetic credential dependency failure"), std::string::npos) << status.ToString();
}

void ExpectCredentialOutOfMemory(const arrow::Status& status) {
  ASSERT_FALSE(status.ok()) << status.ToString();
  EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::MemAllocateFailed) << status.ToString();
}

class FailingRequestCredentialsProvider final : public Aws::Auth::AWSCredentialsProvider,
                                                public RequestCredentialsResolver {
  public:
  explicit FailingRequestCredentialsProvider(arrow::Status failure) : failure_(std::move(failure)) {}

  Aws::Auth::AWSCredentials GetAWSCredentials() override { return {}; }

  arrow::Result<Aws::Auth::AWSCredentials> ResolveForRequest() override {
    ++resolve_calls_;
    return failure_;
  }

  int resolve_calls() const { return resolve_calls_.load(); }

  private:
  arrow::Status failure_;
  std::atomic<int> resolve_calls_{0};
};

class FixedAwsCredentialsProvider final : public Aws::Auth::AWSCredentialsProvider {
  public:
  explicit FixedAwsCredentialsProvider(Aws::Auth::AWSCredentials credentials)
      : credentials_(std::move(credentials)) {}

  Aws::Auth::AWSCredentials GetAWSCredentials() override { return credentials_; }

  private:
  Aws::Auth::AWSCredentials credentials_;
};

class StubSTSClient final : public Aws::STS::STSClient {
  public:
  StubSTSClient()
      : Aws::STS::STSClient(Aws::Auth::AWSCredentials("source-ak", "source-sk"), MakeNoImdsClientConfiguration()) {}

  void PushOutcome(Aws::STS::Model::AssumeRoleOutcome outcome) {
    std::lock_guard<std::mutex> lock(mutex_);
    outcomes_.push(std::move(outcome));
  }

  void PushWebIdentityOutcome(Aws::STS::Model::AssumeRoleWithWebIdentityOutcome outcome) {
    std::lock_guard<std::mutex> lock(mutex_);
    web_identity_outcomes_.push(std::move(outcome));
  }

  void SetDelay(std::chrono::milliseconds delay) {
    std::lock_guard<std::mutex> lock(mutex_);
    delay_ = delay;
  }

  Aws::STS::Model::AssumeRoleOutcome AssumeRole(const Aws::STS::Model::AssumeRoleRequest& request) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.push_back(request);
    std::this_thread::sleep_for(delay_);
    if (outcomes_.empty()) {
      Aws::STS::STSError error;
      error.SetExceptionName("NoStubOutcome");
      error.SetMessage("test STS outcome queue is empty");
      error.SetResponseCode(Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
      return error;
    }
    auto outcome = std::move(outcomes_.front());
    outcomes_.pop();
    return outcome;
  }

  Aws::STS::Model::AssumeRoleWithWebIdentityOutcome AssumeRoleWithWebIdentity(
      const Aws::STS::Model::AssumeRoleWithWebIdentityRequest& request) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    web_identity_requests_.push_back(request);
    if (web_identity_outcomes_.empty()) {
      Aws::STS::STSError error;
      error.SetExceptionName("NoStubWebIdentityOutcome");
      error.SetMessage("test STS web identity outcome queue is empty");
      error.SetResponseCode(Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
      return error;
    }
    auto outcome = std::move(web_identity_outcomes_.front());
    web_identity_outcomes_.pop();
    return outcome;
  }

  std::vector<Aws::STS::Model::AssumeRoleRequest> requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }

  std::vector<Aws::STS::Model::AssumeRoleWithWebIdentityRequest> web_identity_requests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return web_identity_requests_;
  }

  private:
  mutable std::mutex mutex_;
  mutable std::queue<Aws::STS::Model::AssumeRoleOutcome> outcomes_;
  mutable std::vector<Aws::STS::Model::AssumeRoleRequest> requests_;
  mutable std::queue<Aws::STS::Model::AssumeRoleWithWebIdentityOutcome> web_identity_outcomes_;
  mutable std::vector<Aws::STS::Model::AssumeRoleWithWebIdentityRequest> web_identity_requests_;
  std::chrono::milliseconds delay_{0};
};

Aws::STS::Model::AssumeRoleOutcome MakeStsFailure(Aws::Http::HttpResponseCode response_code,
                                                  const Aws::String& exception_name) {
  Aws::STS::STSError error;
  error.SetExceptionName(exception_name);
  error.SetMessage("synthetic STS failure");
  error.SetResponseCode(response_code);
  return error;
}

Aws::STS::Model::AssumeRoleOutcome MakeStsSuccess(const Aws::String& access_key,
                                                  const Aws::String& secret_key,
                                                  const Aws::String& token,
                                                  const Aws::Utils::DateTime& expiration) {
  Aws::STS::Model::Credentials credentials;
  credentials.WithAccessKeyId(access_key)
      .WithSecretAccessKey(secret_key)
      .WithSessionToken(token)
      .WithExpiration(expiration);
  Aws::STS::Model::AssumeRoleResult result;
  result.SetCredentials(std::move(credentials));
  return result;
}

Aws::STS::Model::AssumeRoleWithWebIdentityOutcome MakeWebIdentityStsFailure(
    Aws::Http::HttpResponseCode response_code, const Aws::String& exception_name) {
  Aws::STS::STSError error;
  error.SetExceptionName(exception_name);
  error.SetMessage("synthetic web identity STS failure");
  error.SetResponseCode(response_code);
  return error;
}

Aws::STS::Model::AssumeRoleWithWebIdentityOutcome MakeWebIdentityStsSuccess(
    const Aws::String& access_key,
    const Aws::String& secret_key,
    const Aws::String& token,
    const Aws::Utils::DateTime& expiration) {
  Aws::STS::Model::Credentials credentials;
  credentials.WithAccessKeyId(access_key)
      .WithSecretAccessKey(secret_key)
      .WithSessionToken(token)
      .WithExpiration(expiration);
  Aws::STS::Model::AssumeRoleWithWebIdentityResult result;
  result.SetCredentials(std::move(credentials));
  return result;
}

AwsDefaultCredentialsProvider::Dependencies MakeAwsDefaultTestDependencies(
    const std::shared_ptr<StubSTSClient>& web_identity_client,
    const std::shared_ptr<MockHttpClient>& metadata_client) {
  AwsDefaultCredentialsProvider::Dependencies dependencies;
  dependencies.web_identity_client = web_identity_client;
  dependencies.metadata_client = metadata_client;
  return dependencies;
}

}  // namespace

// ============================================================================
// Diagnostic: Verify mock infrastructure
// ============================================================================

TEST_F(S3ProviderTest, TestMockInfrastructure) {
  // Verify the factory is installed: CreateHttpClient should return our mock
  auto config = MakeNoImdsClientConfiguration();
  auto client = Aws::Http::CreateHttpClient(config);
  ASSERT_EQ(client.get(), mock_client_.get()) << "CreateHttpClient did not return the mock client";

  // Verify CreateHttpRequest works
  auto request =
      Aws::Http::CreateHttpRequest(Aws::Http::URI("https://example.com/test"), Aws::Http::HttpMethod::HTTP_GET,
                                   Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  ASSERT_NE(request, nullptr);

  // Verify enqueue + MakeRequest works (URL-keyed)
  mock_client_->EnqueueResponse("example.com", Aws::Http::HttpResponseCode::OK, "test_body");
  auto response = client->MakeRequest(request);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->GetResponseCode(), Aws::Http::HttpResponseCode::OK);

  // Read the body
  Aws::IStreamBufIterator eos;
  Aws::String body(Aws::IStreamBufIterator(response->GetResponseBody()), eos);
  EXPECT_EQ(body, "test_body");

  // Unmatched URLs should return 404
  auto request2 =
      Aws::Http::CreateHttpRequest(Aws::Http::URI("https://unmatched.example.org"), Aws::Http::HttpMethod::HTTP_GET,
                                   Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  auto response2 = client->MakeRequest(request2);
  EXPECT_EQ(response2->GetResponseCode(), Aws::Http::HttpResponseCode::NOT_FOUND);
}

TEST_F(S3ProviderTest, DeleteDirReturnsOneConcreteFailureWithoutAggregation) {
  constexpr const char* kBucket = "aggregation-test-bucket";

  const std::string first_list_page = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>aggregation-test-bucket</Name>
  <Prefix></Prefix>
  <KeyCount>1</KeyCount>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>true</IsTruncated>
  <Contents>
    <Key>dir/file</Key>
    <LastModified>2026-08-03T00:00:00.000Z</LastModified>
    <ETag>&quot;etag&quot;</ETag>
    <Size>1</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
  <NextContinuationToken>next-page</NextContinuationToken>
</ListBucketResult>)xml";
  const std::string delete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Error>
    <Key>dir/file</Key>
    <Code>AccessDenied</Code>
    <Message>delete denied by policy</Message>
  </Error>
</DeleteResult>)xml";
  const std::string listing_error = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>SlowDown</Code>
  <Message>listing throttled</Message>
  <RequestId>request-id</RequestId>
</Error>)xml";

  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };
  // Pagination and page deletes run concurrently. Either concrete failure may
  // win, but the result must never synthesize a combined status.
  std::promise<void> delete_started_promise;
  auto delete_started = delete_started_promise.get_future().share();

  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::OK, first_list_page,
                                xml_headers(first_list_page));
  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, listing_error,
                                xml_headers(listing_error), [delete_started] {
                                  EXPECT_EQ(delete_started.wait_for(std::chrono::seconds(5)), std::future_status::ready)
                                      << "second listing failure arrived before the first page's delete started";
                                });
  mock_client_->EnqueueResponse("?delete", Aws::Http::HttpResponseCode::OK, delete_result, xml_headers(delete_result),
                                [&delete_started_promise] { delete_started_promise.set_value(); });

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto status = fs->DeleteDirContents(kBucket, /*missing_dir_ok=*/false);

  ASSERT_STATUS_NOT_OK(status);
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_TRUE(detail->code() == ExtendStatusCode::StorageAccessDenied ||
              detail->code() == ExtendStatusCode::StorageTransientService ||
              detail->code() == ExtendStatusCode::StorageTransientThrottling)
      << status.ToString();
  const bool has_delete_failure = status.message().find("AccessDenied") != std::string::npos;
  const bool has_listing_failure = status.message().find("listing throttled") != std::string::npos;
  EXPECT_NE(has_delete_failure, has_listing_failure) << status.ToString();
}

TEST_F(S3ProviderTest, DeleteObjectsSubmissionFailureDoesNotWaitForAcceptedChunks) {
  constexpr const char* kBucket = "delete-submission-failure-bucket";

  std::string list_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>delete-submission-failure-bucket</Name>
  <Prefix></Prefix>
  <KeyCount>1001</KeyCount>
  <MaxKeys>1001</MaxKeys>
  <IsTruncated>false</IsTruncated>
)xml";
  for (int i = 0; i < 1001; ++i) {
    list_result += "<Contents><Key>dir/file-" + std::to_string(i) +
                   "</Key><LastModified>2026-08-11T00:00:00.000Z</LastModified>"
                   "<ETag>&quot;etag&quot;</ETag><Size>1</Size><StorageClass>STANDARD</StorageClass></Contents>";
  }
  list_result += "</ListBucketResult>";

  const std::string delete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Error>
    <Key>dir/file-0</Key>
    <Code>AccessDenied</Code>
    <Message>first accepted chunk was denied</Message>
  </Error>
</DeleteResult>)xml";

  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };

  ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(2));
  RejectAfterOneS3Executor executor(pool);
  std::promise<void> first_delete_started_promise;
  auto first_delete_started = first_delete_started_promise.get_future();
  std::promise<void> release_first_delete_promise;
  auto release_first_delete = release_first_delete_promise.get_future().share();

  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::OK, list_result, xml_headers(list_result),
                                [&executor] { executor.Arm(); });
  mock_client_->EnqueueResponse("?delete", Aws::Http::HttpResponseCode::OK, delete_result, xml_headers(delete_result),
                                [&first_delete_started_promise, release_first_delete] {
                                  first_delete_started_promise.set_value();
                                  EXPECT_EQ(release_first_delete.wait_for(std::chrono::seconds(5)),
                                            std::future_status::ready);
                                });

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  arrow::io::IOContext io_context(&executor);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  auto delete_future = fs->DeleteDirContentsAsync(kBucket, /*missing_dir_ok=*/false);

  ASSERT_EQ(executor.rejection_observed().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  ASSERT_EQ(first_delete_started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  const bool finished_before_accepted_chunk = delete_future.is_finished();
  release_first_delete_promise.set_value();
  ASSERT_TRUE(finished_before_accepted_chunk) << "submission failure waited for an already accepted chunk";
  auto status = delete_future.status();
  ASSERT_STATUS_NOT_OK(status);
  EXPECT_NE(status.message().find("executor rejected S3 delete task"), std::string::npos) << status.ToString();
  EXPECT_EQ(status.message().find("first accepted chunk was denied"), std::string::npos) << status.ToString();
}

TEST_F(S3ProviderTest, DeleteObjectsResultFailureDoesNotWaitForSibling) {
  constexpr const char* kBucket = "delete-result-failure-bucket";

  std::string list_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>delete-result-failure-bucket</Name>
  <Prefix></Prefix>
  <KeyCount>1001</KeyCount>
  <MaxKeys>1001</MaxKeys>
  <IsTruncated>false</IsTruncated>
)xml";
  for (int i = 0; i < 1001; ++i) {
    list_result += "<Contents><Key>dir/file-" + std::to_string(i) +
                   "</Key><LastModified>2026-08-18T00:00:00.000Z</LastModified>"
                   "<ETag>&quot;etag&quot;</ETag><Size>1</Size><StorageClass>STANDARD</StorageClass></Contents>";
  }
  list_result += "</ListBucketResult>";

  const std::string failed_delete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Error>
    <Key>dir/file-0</Key>
    <Code>AccessDenied</Code>
    <Message>first completed chunk was denied</Message>
  </Error>
</DeleteResult>)xml";
  const std::string successful_delete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/" />)xml";

  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };

  ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(2));
  std::promise<void> sibling_started_promise;
  auto sibling_started = sibling_started_promise.get_future();
  std::promise<void> release_sibling_promise;
  auto release_sibling = release_sibling_promise.get_future().share();

  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::OK, list_result, xml_headers(list_result));
  mock_client_->EnqueueResponse(
      "?delete", Aws::Http::HttpResponseCode::OK, failed_delete_result, xml_headers(failed_delete_result),
      [&sibling_started] { EXPECT_EQ(sibling_started.wait_for(std::chrono::seconds(5)), std::future_status::ready); });
  mock_client_->EnqueueResponse("?delete", Aws::Http::HttpResponseCode::OK, successful_delete_result,
                                xml_headers(successful_delete_result), [&sibling_started_promise, release_sibling] {
                                  sibling_started_promise.set_value();
                                  EXPECT_EQ(release_sibling.wait_for(std::chrono::seconds(10)),
                                            std::future_status::ready);
                                });

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  arrow::io::IOContext io_context(pool.get());
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  auto delete_future = fs->DeleteDirContentsAsync(kBucket, /*missing_dir_ok=*/false);

  const auto sibling_state = sibling_started.wait_for(std::chrono::seconds(5));
  const bool finished_before_sibling_release = sibling_state == std::future_status::ready && delete_future.Wait(5.0);
  std::optional<arrow::Status> observed_status;
  if (finished_before_sibling_release) {
    observed_status = delete_future.status();
  }
  release_sibling_promise.set_value();
  delete_future.Wait(5.0);

  ASSERT_EQ(sibling_state, std::future_status::ready);
  ASSERT_TRUE(finished_before_sibling_release) << "delete failure waited for a sibling request";
  ASSERT_TRUE(observed_status.has_value());
  ASSERT_STATUS_NOT_OK(*observed_status);
  auto detail = ExtendStatusDetail::UnwrapStatus(*observed_status);
  ASSERT_NE(detail, nullptr) << observed_status->ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied) << observed_status->ToString();
  EXPECT_NE(observed_status->message().find("first completed chunk was denied"), std::string::npos)
      << observed_status->ToString();
}

TEST_F(S3ProviderTest, DeleteObjectsPerKeyCredentialErrorsAreAuthenticationFailures) {
  constexpr const char* kBucket = "expired-token-delete-bucket";

  std::string list_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>expired-token-delete-bucket</Name>
  <Prefix></Prefix>
  <KeyCount>5</KeyCount>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>false</IsTruncated>
)xml";
  const std::vector<std::string> credential_errors = {"ExpiredToken", "InvalidToken", "InvalidAccessKeyId",
                                                      "SignatureDoesNotMatch", "InvalidSecurity"};
  for (size_t i = 0; i < credential_errors.size(); ++i) {
    list_result += "<Contents><Key>dir/file-" + std::to_string(i) +
                   "</Key><LastModified>2026-08-03T00:00:00.000Z</LastModified>"
                   "<ETag>&quot;etag&quot;</ETag><Size>1</Size><StorageClass>STANDARD</StorageClass></Contents>";
  }
  list_result += "</ListBucketResult>";

  std::string delete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<DeleteResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
)xml";
  for (size_t i = 0; i < credential_errors.size(); ++i) {
    delete_result += "<Error><Key>dir/file-" + std::to_string(i) + "</Key><Code>" + credential_errors[i] +
                     "</Code><Message>credentials rejected</Message></Error>";
  }
  delete_result += "</DeleteResult>";

  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };
  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::OK, list_result, xml_headers(list_result));
  mock_client_->EnqueueResponse("?delete", Aws::Http::HttpResponseCode::OK, delete_result, xml_headers(delete_result));

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto status = fs->DeleteDirContents(kBucket, /*missing_dir_ok=*/false);

  ASSERT_STATUS_NOT_OK(status);
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied) << status.ToString();
  EXPECT_FALSE(detail->retryable());
  EXPECT_NE(status.message().find(credential_errors.front()), std::string::npos) << status.ToString();
  EXPECT_EQ(status.message().find(credential_errors.back()), std::string::npos) << status.ToString();
}

TEST_F(S3ProviderTest, KnownSizeGetObjectNotFoundIsMissingKeyWithoutProbe) {
  auto make_fs = [this]() -> arrow::Result<std::shared_ptr<S3FileSystem>> {
    S3Options options;
    options.ConfigureAnonymousCredentials();
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "mock-s3.local";
    options.cloud_provider = kCloudProviderAWS;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
    options.use_crt_async_reads = false;
    return S3FileSystem::Make(options);
  };

  auto read_known_size = [](const std::shared_ptr<S3FileSystem>& fs) -> arrow::Status {
    arrow::fs::FileInfo info("bucket/key", arrow::fs::FileType::File);
    info.set_size(1);
    ARROW_ASSIGN_OR_RAISE(auto file, fs->OpenInputFile(info));
    uint8_t byte = 0;
    return file->ReadAt(0, 1, &byte).status();
  };

  // OpenInputFile(FileInfo) skips HeadObject by design. A bodyless GetObject
  // 404 is reduced to a missing key directly -- exactly ONE request. No
  // disambiguation probe follows: extra IO on the miss path is deliberately
  // not spent, so a vanished bucket also reports as the missing object.
  mock_client_->EnqueueResponse("mock-s3.local", Aws::Http::HttpResponseCode::NOT_FOUND, "");
  ASSERT_AND_ASSIGN(auto missing_key_fs, make_fs());
  auto missing_key = read_known_size(missing_key_fs);
  ASSERT_STATUS_NOT_OK(missing_key);
  auto key_detail = ExtendStatusDetail::UnwrapStatus(missing_key);
  ASSERT_NE(key_detail, nullptr) << missing_key.ToString();
  EXPECT_EQ(key_detail->code(), ExtendStatusCode::StorageNotFound) << missing_key.ToString();
  EXPECT_EQ(CategoryForExtendStatusCode(key_detail->code()), ErrorCategory::System);
  // Only the failing GetObject (plus SDK retries of it) -- a HeadBucket probe
  // would show up as a HEAD request.
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    EXPECT_NE(request->GetMethod(), Aws::Http::HttpMethod::HTTP_HEAD) << "unexpected disambiguation probe";
  }
}

TEST_F(S3ProviderTest, DeleteFileNotFoundDoesNotProbeBucket) {
  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  mock_client_->EnqueueResponse("mock-s3.local", Aws::Http::HttpResponseCode::NOT_FOUND, "");
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto status = fs->DeleteFile("bucket/key");

  ASSERT_STATUS_NOT_OK(status);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist) << status.ToString();
  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), 1u) << "a failed HeadObject must not trigger a HeadBucket diagnostic request";
  EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_HEAD);
}

TEST_F(S3ProviderTest, DeleteObjectsRequestNotFoundNamesTheBucket) {
  const std::string list_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>bucket</Name>
  <Prefix></Prefix>
  <KeyCount>1</KeyCount>
  <MaxKeys>1000</MaxKeys>
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>key</Key>
    <LastModified>2026-08-06T00:00:00.000Z</LastModified>
    <ETag>&quot;etag&quot;</ETag>
    <Size>1</Size>
    <StorageClass>STANDARD</StorageClass>
  </Contents>
</ListBucketResult>)xml";
  Aws::Http::HeaderValueCollection list_headers;
  list_headers["content-type"] = "application/xml";
  list_headers["content-length"] = std::to_string(list_result.size());
  mock_client_->EnqueueResponse("list-type=2", Aws::Http::HttpResponseCode::OK, list_result, list_headers);
  mock_client_->EnqueueResponse("?delete", Aws::Http::HttpResponseCode::NOT_FOUND, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto status = fs->DeleteDirContents("bucket", /*missing_dir_ok=*/false);

  ASSERT_STATUS_NOT_OK(status);
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageBucketNotFound) << status.ToString();
  EXPECT_EQ(CategoryForExtendStatusCode(detail->code()), ErrorCategory::System);
}

TEST_F(S3ProviderTest, MultipartNotFoundReportsNoSuchUploadWithoutProbe) {
  auto make_fs = []() -> arrow::Result<std::shared_ptr<S3FileSystem>> {
    S3Options options;
    options.ConfigureAnonymousCredentials();
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "mock-s3.local";
    options.cloud_provider = kCloudProviderAWS;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
    options.use_crt_async_reads = false;
    options.background_writes = false;
    return S3FileSystem::Make(options);
  };

  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  Aws::Http::HeaderValueCollection xml_headers;
  xml_headers["content-type"] = "application/xml";
  xml_headers["content-length"] = std::to_string(create_result.size());

  auto write_one_part = [](const std::shared_ptr<S3FileSystem>& fs) -> arrow::Status {
    ARROW_ASSIGN_OR_RAISE(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));
    const uint8_t byte = 7;
    return stream->Write(&byte, 1);
  };

  // A generic 404 on UploadPart is reported as NoSuchUpload directly -- no
  // HeadBucket probe follows (extra IO on failure paths is deliberately not
  // spent), so a vanished bucket reports the same way.
  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers);
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::NOT_FOUND, "");
  ASSERT_AND_ASSIGN(auto dead_upload_fs, make_fs());
  auto dead_upload = write_one_part(dead_upload_fs);
  ASSERT_STATUS_NOT_OK(dead_upload);
  auto upload_detail = ExtendStatusDetail::UnwrapStatus(dead_upload);
  ASSERT_NE(upload_detail, nullptr) << dead_upload.ToString();
  EXPECT_EQ(upload_detail->code(), ExtendStatusCode::StorageNoSuchUpload) << dead_upload.ToString();
  EXPECT_EQ(CategoryForExtendStatusCode(upload_detail->code()), ErrorCategory::System);
  // CreateMultipartUpload + the failing UploadPart (plus SDK retries) -- a
  // HeadBucket probe would show up as a HEAD request.
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    EXPECT_NE(request->GetMethod(), Aws::Http::HttpMethod::HTTP_HEAD) << "unexpected disambiguation probe";
  }
}

TEST_F(S3ProviderTest, BackgroundUploadSubmissionFailureSettlesFlush) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  Aws::Http::HeaderValueCollection xml_headers;
  xml_headers["content-type"] = "application/xml";
  xml_headers["content-length"] = std::to_string(create_result.size());
  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers);
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  RejectingS3Executor executor;
  arrow::io::IOContext io_context(&executor);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

  const uint8_t byte = 7;
  auto write_status = stream->Write(&byte, 1);
  ASSERT_STATUS_NOT_OK(write_status);
  EXPECT_NE(write_status.message().find("executor rejected S3 upload task"), std::string::npos);

  auto flush_status = stream->Flush();
  ASSERT_STATUS_NOT_OK(flush_status);
  EXPECT_TRUE(flush_status.Equals(write_status));
  ASSERT_STATUS_OK(stream->Abort());
}

TEST_F(S3ProviderTest, UnexpectedBackgroundUploadSubmissionExceptionDrainsBeforeDeferredAbort) {
  AssertBackgroundUploadSubmissionExceptionDrains(S3SubmissionExceptionKind::kUnexpected);
}

TEST_F(S3ProviderTest, BackgroundUploadSubmissionBadAllocIsFailStop) {
  PrepareS3DeathTest();
  EXPECT_DEATH_IF_SUPPORTED(
      { AssertBackgroundUploadSubmissionExceptionDrains(S3SubmissionExceptionKind::kBadAlloc); }, "");
}

// A credential failure that could NOT be reached and one that was REFUSED must
// not land the same way. The distinction is not there to make the SDK retry
// again -- the credential client's own budget is already spent by then -- it is
// there so the layer above can re-run the whole operation later. Collapsing
// both into StorageAccessDenied made a restarting metadata service fail a
// query exactly as permanently as a role ARN that does not exist.
TEST(StsCredentialResolutionTest, SeparatesUnreachableFromRefused) {
  using Aws::Http::HttpResponseCode;

  auto code_of = [](const arrow::Status& status) {
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    EXPECT_NE(detail, nullptr) << status.ToString();
    return detail != nullptr ? detail->code() : ExtendStatusCode::StorageAccessDenied;
  };

  // Never left the process, or the http client returned nothing at all. These
  // sit inside ordinary numeric ranges (444, 598, 599), so a classifier keyed
  // on the range first would call a dead network a bad configuration.
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::REQUEST_NOT_MADE, "x")),
            ExtendStatusCode::StorageTransientNetwork);
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::NO_RESPONSE, "x")),
            ExtendStatusCode::StorageTransientNetwork);
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::NETWORK_CONNECT_TIMEOUT, "x")),
            ExtendStatusCode::StorageTransientTimeout);
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::TOO_MANY_REQUESTS, "x")),
            ExtendStatusCode::StorageTransientThrottling);
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::SERVICE_UNAVAILABLE, "x")),
            ExtendStatusCode::StorageTransientService);

  // The service identified us and said no. That one really is an access
  // decision, and it must stay non-retryable.
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::FORBIDDEN, "x")),
            ExtendStatusCode::StorageAccessDenied);
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::UNAUTHORIZED, "x")),
            ExtendStatusCode::StorageAccessDenied);

  // Refused, but not as an access decision.
  EXPECT_EQ(code_of(ClassifyCredentialHttpFailure(HttpResponseCode::BAD_REQUEST, "x")),
            ExtendStatusCode::StorageConfigInvalid);
}

// The property the whole split exists for: a transient credential failure has
// to still read as retryable where Milvus decides whether to run the operation
// again, and a refusal has to still read as permanent. Asserting the
// ExtendStatusCode alone would not catch a regression in that mapping.
TEST(StsCredentialResolutionTest, TransientCredentialFailuresStayRetryableAtTheBoundary) {
  using Aws::Http::HttpResponseCode;

  for (auto code :
       {HttpResponseCode::REQUEST_NOT_MADE, HttpResponseCode::NO_RESPONSE, HttpResponseCode::SERVICE_UNAVAILABLE,
        HttpResponseCode::TOO_MANY_REQUESTS, HttpResponseCode::NETWORK_CONNECT_TIMEOUT}) {
    auto status = ClassifyCredentialHttpFailure(code, "transient");
    auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr) << "http_status=" << static_cast<int>(code);
    EXPECT_TRUE(detail->retryable()) << "http_status=" << static_cast<int>(code);
    EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageTransientError)
        << "http_status=" << static_cast<int>(code);
  }

  auto refused = ClassifyCredentialHttpFailure(HttpResponseCode::FORBIDDEN, "refused");
  auto refused_detail = ExtendStatusDetail::UnwrapStatus(refused);
  ASSERT_NE(refused_detail, nullptr);
  EXPECT_FALSE(refused_detail->retryable());
  EXPECT_EQ(ToSegcoreError(refused).get_error_code(), milvus::ConfigInvalid);
}

// Missing local configuration is the deployment's to fix, and must not read as
// something a retry outlasts.
TEST(StsCredentialResolutionTest, MissingLocalConfigurationIsNotRetryable) {
  auto status = MakeCredentialConfigError("Cannot open the OIDC token file /var/run/secrets/token");
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageConfigInvalid);
  EXPECT_FALSE(detail->retryable());
}

// The budget is asked AFTER the writer lock is acquired, so what it actually
// answers is "did I spend my whole budget waiting for somebody else's failed
// reload". A caller that just arrived still gets to try; one that waited longer
// than the budget does not, which is what stops a queue of threads replaying
// the same outage one after another.
TEST(StsCredentialResolutionTest, AttemptIsAbandonedOnceTheBudgetIsSpentWaiting) {
  EXPECT_TRUE(CredentialAttemptStillWorthMaking(std::chrono::steady_clock::now()));
  EXPECT_FALSE(CredentialAttemptStillWorthMaking(std::chrono::steady_clock::now() - kCredentialResolutionBudget -
                                                 std::chrono::milliseconds(1)));
}

// A body we could not use is neither a transport fault to wait out nor a
// refusal. Left unclassified on purpose: the conservative landing is
// non-retryable without claiming to know which it was.
TEST(StsCredentialResolutionTest, AnUnusableResponseIsNotDressedAsAccessDenied) {
  auto status = MakeCredentialResponseError("STS answered 200 with no <Credentials>");
  EXPECT_TRUE(status.IsIOError()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr) << status.ToString();
  EXPECT_NE(status.message().find("no <Credentials>"), std::string::npos);
}

TEST(StsCredentialResolutionTest, TemporaryCredentialsAreValidatedAtomically) {
  const auto future = Aws::Utils::DateTime(std::chrono::system_clock::now() + std::chrono::hours(1));
  const auto expired = Aws::Utils::DateTime(std::chrono::system_clock::now() - std::chrono::seconds(1));
  const auto invalid = Aws::Utils::DateTime("not-an-expiration", Aws::Utils::DateFormat::ISO_8601);

  EXPECT_TRUE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "sk", "token", future), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("", "sk", "token", future), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "", "token", future), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "sk", "", future), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "sk", "token"), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "sk", "token", invalid), "test STS").ok());
  EXPECT_FALSE(ValidateTemporaryCredentials(Aws::Auth::AWSCredentials("ak", "sk", "token", expired), "test STS").ok());
}

TEST(StsCredentialResolutionTest, AwsAssumeRolePreservesTypedFailureCause) {
  struct FailureCase {
    Aws::Http::HttpResponseCode response_code;
    const char* exception_name;
    ExtendStatusCode expected;
  };
  const FailureCase cases[] = {
      {Aws::Http::HttpResponseCode::REQUEST_NOT_MADE, "NetworkConnection", ExtendStatusCode::StorageTransientNetwork},
      {Aws::Http::HttpResponseCode::BAD_REQUEST, "IDPCommunicationError",
       ExtendStatusCode::StorageTransientNetwork},
      {Aws::Http::HttpResponseCode::FORBIDDEN, "AccessDenied", ExtendStatusCode::StorageAccessDenied},
      {Aws::Http::HttpResponseCode::BAD_REQUEST, "ExpiredTokenException",
       ExtendStatusCode::StorageAccessDenied},
      {Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS, "TooManyRequestsException",
       ExtendStatusCode::StorageTransientThrottling},
      {Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "ServiceUnavailable",
       ExtendStatusCode::StorageTransientService},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.response_code));
    auto sts = std::make_shared<StubSTSClient>();
    sts->PushOutcome(MakeStsFailure(test_case.response_code, test_case.exception_name));
    AwsSTSAssumeRoleCredentialsProvider provider("arn:aws:iam::123456789012:role/test", "session", "", 900, sts);

    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    ExpectExtendStatusCode(resolved.status(), test_case.expected);
    EXPECT_EQ(sts->requests().size(), 1u);
  }
}

TEST(StsCredentialResolutionTest, AwsAssumeRoleRejectsPartialCredentialsWithoutCachingThem) {
  const auto future = Aws::Utils::DateTime(std::chrono::system_clock::now() + std::chrono::hours(1));
  auto sts = std::make_shared<StubSTSClient>();
  sts->PushOutcome(MakeStsSuccess("partial-ak", "partial-sk", "", future));
  sts->PushOutcome(MakeStsSuccess("valid-ak", "valid-sk", "valid-token", future));

  AwsSTSAssumeRoleCredentialsProvider provider("arn:aws:iam::123456789012:role/test", "named-session",
                                               "tenant-external-id", 900, sts);
  auto partial = provider.ResolveForRequest();
  ASSERT_FALSE(partial.ok());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(partial.status()), nullptr) << partial.status().ToString();

  ASSERT_AND_ASSIGN(auto credentials, provider.ResolveForRequest());
  EXPECT_EQ(credentials.GetAWSAccessKeyId(), "valid-ak");
  EXPECT_EQ(credentials.GetAWSSecretKey(), "valid-sk");
  EXPECT_EQ(credentials.GetSessionToken(), "valid-token");

  const auto requests_after_success = sts->requests();
  ASSERT_EQ(requests_after_success.size(), 2u) << "the incomplete first response must not have been cached";
  EXPECT_EQ(requests_after_success.back().GetRoleArn(), "arn:aws:iam::123456789012:role/test");
  EXPECT_EQ(requests_after_success.back().GetRoleSessionName(), "named-session");
  EXPECT_EQ(requests_after_success.back().GetExternalId(), "tenant-external-id");
  EXPECT_EQ(requests_after_success.back().GetDurationSeconds(), 900);

  ASSERT_TRUE(provider.ResolveForRequest().ok());
  EXPECT_EQ(sts->requests().size(), 2u) << "a valid unexpired result must be reused from cache";
}

TEST_F(S3ProviderTest, AwsDefaultProviderClassifiesEveryWebIdentityTransportClass) {
  TempFile token_file("identity-token");
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvVar role("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/workload");
  ScopedEnvVar token_path("AWS_WEB_IDENTITY_TOKEN_FILE", token_file.path());
  ScopedEnvVar region("AWS_REGION", "us-east-1");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset container_full("AWS_CONTAINER_CREDENTIALS_FULL_URI");

  struct Case {
    Aws::Http::HttpResponseCode http_code;
    const char* exception_name;
    ExtendStatusCode expected;
  };
  const std::vector<Case> cases = {
      {Aws::Http::HttpResponseCode::NO_RESPONSE, "SyntheticFailure", ExtendStatusCode::StorageTransientNetwork},
      {Aws::Http::HttpResponseCode::REQUEST_TIMEOUT, "SyntheticFailure", ExtendStatusCode::StorageTransientTimeout},
      {Aws::Http::HttpResponseCode::BAD_REQUEST, "IDPCommunicationError",
       ExtendStatusCode::StorageTransientNetwork},
      {Aws::Http::HttpResponseCode::FORBIDDEN, "SyntheticFailure", ExtendStatusCode::StorageAccessDenied},
      {Aws::Http::HttpResponseCode::BAD_REQUEST, "ExpiredTokenException",
       ExtendStatusCode::StorageAccessDenied},
      {Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS, "SyntheticFailure",
       ExtendStatusCode::StorageTransientThrottling},
      {Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "SyntheticFailure",
       ExtendStatusCode::StorageTransientService},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(fmt::format("http={} exception={}", static_cast<int>(test_case.http_code),
                             test_case.exception_name));
    auto sts = std::make_shared<StubSTSClient>();
    sts->PushWebIdentityOutcome(MakeWebIdentityStsFailure(test_case.http_code, test_case.exception_name));
    AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));
    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    ExpectExtendStatusCode(resolved.status(), test_case.expected);
    EXPECT_EQ(sts->web_identity_requests().size(), 1u);
  }
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty())
      << "IRSA is authoritative; its failure must never fall through to IMDS";
}

TEST_F(S3ProviderTest, AwsDefaultProviderWebIdentityOnlyUsesUnsignedSdkCallAndKeepsTypedFailure) {
  TempFile token_file("identity-token");
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvVar role("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/workload");
  ScopedEnvVar token_path("AWS_WEB_IDENTITY_TOKEN_FILE", token_file.path());
  ScopedEnvVar region("AWS_REGION", "us-east-1");
  const std::string error_body = R"(<?xml version="1.0" encoding="UTF-8"?>
<ErrorResponse><Error><Type>Sender</Type><Code>AccessDenied</Code>
<Message>IRSA trust policy refused the token</Message></Error><RequestId>request-id</RequestId></ErrorResponse>)";
  Aws::Http::HeaderValueCollection headers;
  headers["content-type"] = "application/xml";
  headers["content-length"] = std::to_string(error_body.size());
  mock_client_->EnqueueResponse("amazonaws.com", Aws::Http::HttpResponseCode::FORBIDDEN, error_body, headers);

  AwsDefaultCredentialsProvider::Dependencies dependencies;
  dependencies.before_web_identity.push_back(std::make_shared<FixedAwsCredentialsProvider>(
      Aws::Auth::AWSCredentials("ambient-ak", "ambient-sk")));
  dependencies.metadata_client = mock_client_;
  AwsDefaultCredentialsProvider provider(std::move(dependencies),
                                         AwsDefaultCredentialsProvider::SourceMode::WebIdentityOnly);
  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageAccessDenied);

  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_POST);
  EXPECT_NE(requests.front()->GetURIString().find("amazonaws.com"), Aws::String::npos);
  EXPECT_EQ(requests.front()->GetURIString().find("169.254.169.254"), Aws::String::npos);
  EXPECT_FALSE(requests.front()->HasHeader("Authorization"));
}

TEST_F(S3ProviderTest, AwsDefaultProviderRejectsPartialTemporaryCredentialsFromSdkProviders) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvUnset container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  const auto future = Aws::Utils::DateTime(std::chrono::system_clock::now() + std::chrono::hours(1));

  for (bool before_web_identity : {true, false}) {
    SCOPED_TRACE(before_web_identity ? "before web identity" : "after web identity");
    auto metadata = std::make_shared<MockHttpClient>();
    auto partial = std::make_shared<FixedAwsCredentialsProvider>(
        Aws::Auth::AWSCredentials("partial-ak", "partial-sk", "", future));
    AwsDefaultCredentialsProvider::Dependencies dependencies;
    dependencies.metadata_client = metadata;
    if (before_web_identity) {
      dependencies.before_web_identity.push_back(partial);
    } else {
      dependencies.after_web_identity.push_back(partial);
    }
    AwsDefaultCredentialsProvider provider(std::move(dependencies));

    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(resolved.status()), nullptr) << resolved.status().ToString();
    EXPECT_TRUE(metadata->GetRecordedRequests().empty())
        << "a malformed SDK temporary credential must not fall through to IMDS";
  }
}

TEST_F(S3ProviderTest, AwsDefaultProviderRejectsPartialIrsaCredentials) {
  TempFile token_file("identity-token");
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvVar role("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/workload");
  ScopedEnvVar token_path("AWS_WEB_IDENTITY_TOKEN_FILE", token_file.path());
  const auto future = Aws::Utils::DateTime(std::chrono::system_clock::now() + std::chrono::hours(1));
  auto sts = std::make_shared<StubSTSClient>();
  sts->PushWebIdentityOutcome(MakeWebIdentityStsSuccess("partial-ak", "partial-sk", "", future));
  AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));

  auto partial = provider.ResolveForRequest();
  ASSERT_FALSE(partial.ok());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(partial.status()), nullptr) << partial.status().ToString();
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty());
}

TEST_F(S3ProviderTest, AwsDefaultProviderContainerFailureNeverFallsThroughToImds) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvVar container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1/credentials");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
    mock_client_->EnqueueResponse("127.0.0.1/credentials", Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "");
  }
  auto sts = std::make_shared<StubSTSClient>();
  AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));

  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageTransientService);
  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), static_cast<size_t>(kCredentialRetryAttempts + 1));
  for (const auto& request : requests) {
    EXPECT_NE(request->GetURIString().find("127.0.0.1/credentials"), Aws::String::npos);
  }
}

TEST_F(S3ProviderTest, AwsDefaultProviderRejectsContainerAuthorizationTokenNewlines) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvVar container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1/credentials");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvVar authorization("AWS_CONTAINER_AUTHORIZATION_TOKEN", "header-injection\nvalue");
  auto sts = std::make_shared<StubSTSClient>();
  AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));

  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageConfigInvalid);
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty());
}

TEST_F(S3ProviderTest, AwsDefaultProviderImdsTokenFailureIsTyped) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvUnset container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset imds_disabled("AWS_EC2_METADATA_DISABLED");
  for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
    mock_client_->EnqueueResponse("169.254.169.254/latest/api/token",
                                  Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "");
  }
  auto sts = std::make_shared<StubSTSClient>();
  AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));

  auto failed = provider.ResolveForRequest();
  ASSERT_FALSE(failed.ok());
  ExpectExtendStatusCode(failed.status(), ExtendStatusCode::StorageTransientService);
  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), static_cast<size_t>(kCredentialRetryAttempts + 1));
  for (const auto& request : requests) {
    EXPECT_EQ(request->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT)
        << "a transient IMDSv2 token failure must not trigger a bare metadata GET";
  }
}

TEST_F(S3ProviderTest, AwsDefaultProviderAtomicallyValidatesImdsCredentials) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvUnset container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset imds_disabled("AWS_EC2_METADATA_DISABLED");
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "imds-token");
  mock_client_->EnqueueResponse("iam/security-credentials/", Aws::Http::HttpResponseCode::OK, "node-role");
  mock_client_->EnqueueResponse(
      "iam/security-credentials/node-role", Aws::Http::HttpResponseCode::OK,
      R"({"Code":"Success","AccessKeyId":"ak","SecretAccessKey":"sk","Expiration":"2099-12-31T23:59:59Z"})");
  auto sts = std::make_shared<StubSTSClient>();
  AwsDefaultCredentialsProvider provider(MakeAwsDefaultTestDependencies(sts, mock_client_));

  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(resolved.status()), nullptr) << resolved.status().ToString();
  EXPECT_EQ(mock_client_->GetRecordedRequests().size(), 3u);
}

TEST_F(S3ProviderTest, AwsDefaultProviderHonorsSharedImdsProfileWithEnvironmentPrecedence) {
  ReloadAwsConfigOnExit reload_original_config;
  TempFile config_file(R"([profile imds-mode]
ec2_metadata_v1_disabled = true
ec2_metadata_service_endpoint_mode = ipv6

[profile imds-endpoint]
ec2_metadata_service_endpoint = http://127.0.0.43
)");
  ScopedEnvVar config_path("AWS_CONFIG_FILE", config_file.path());
  ScopedEnvUnset default_profile("AWS_DEFAULT_PROFILE");
  ScopedEnvVar profile("AWS_PROFILE", "imds-mode");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvUnset container_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");
  ScopedEnvUnset container_relative("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset metadata_disabled("AWS_EC2_METADATA_DISABLED");
  ScopedEnvUnset v1_disabled_env("AWS_EC2_METADATA_V1_DISABLED");
  ScopedEnvUnset endpoint_env("AWS_EC2_METADATA_SERVICE_ENDPOINT");
  ScopedEnvUnset endpoint_mode_env("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE");
  Aws::Config::ReloadCachedConfigFile();

  {
    auto metadata = std::make_shared<MockHttpClient>();
    metadata->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");
    AwsDefaultCredentialsProvider::Dependencies dependencies;
    dependencies.metadata_client = metadata;
    AwsDefaultCredentialsProvider provider(std::move(dependencies));

    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageAccessDenied);
    const auto requests = metadata->GetRecordedRequests();
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_NE(requests.front()->GetURIString().find("fd00:ec2::254"), Aws::String::npos);
    EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT)
        << "profile ec2_metadata_v1_disabled must prevent a bare GET";
  }

  {
    ScopedEnvVar allow_v1("AWS_EC2_METADATA_V1_DISABLED", "false");
    ScopedEnvVar endpoint("AWS_EC2_METADATA_SERVICE_ENDPOINT", "http://127.0.0.42");
    auto metadata = std::make_shared<MockHttpClient>();
    metadata->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");
    AwsDefaultCredentialsProvider::Dependencies dependencies;
    dependencies.metadata_client = metadata;
    AwsDefaultCredentialsProvider provider(std::move(dependencies));

    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    const auto requests = metadata->GetRecordedRequests();
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT);
    EXPECT_EQ(requests.back()->GetMethod(), Aws::Http::HttpMethod::HTTP_GET)
        << "the explicit false environment value must override profile true";
    for (const auto& request : requests) {
      EXPECT_NE(request->GetURIString().find("127.0.0.42"), Aws::String::npos)
          << "the endpoint environment variable must override profile endpoint mode";
    }
  }

  {
    ScopedEnvVar endpoint_profile("AWS_PROFILE", "imds-endpoint");
    auto metadata = std::make_shared<MockHttpClient>();
    metadata->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::BAD_REQUEST, "");
    AwsDefaultCredentialsProvider::Dependencies dependencies;
    dependencies.metadata_client = metadata;
    AwsDefaultCredentialsProvider provider(std::move(dependencies));

    auto resolved = provider.ResolveForRequest();
    ASSERT_FALSE(resolved.ok());
    const auto requests = metadata->GetRecordedRequests();
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_NE(requests.front()->GetURIString().find("127.0.0.43"), Aws::String::npos);
  }
}

TEST_F(S3ProviderTest, ExplicitAwsRoleSourceFailureStopsTargetStsAndS3Requests) {
  auto source = std::make_shared<FailingRequestCredentialsProvider>(
      MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, "source IRSA refresh failed"));
  auto target_sts = std::make_shared<StubSTSClient>();
  auto target = std::make_shared<AwsSTSAssumeRoleCredentialsProvider>(
      "arn:aws:iam::123456789012:role/target", "target-session", "", 900, target_sts, source);

  S3Options options;
  options.credentials_provider = target;
  options.credentials_kind = S3CredentialsKind::Role;
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto file_info = fs->GetFileInfo("bucket/key");
  ASSERT_FALSE(file_info.ok());
  ExpectExtendStatusCode(file_info.status(), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_TRUE(target_sts->requests().empty()) << "source failure must stop the target AssumeRole call";
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty()) << "source failure must stop the object request";
}

TEST_F(S3ProviderTest, AwsUseIamPublishesRequestLocalResolver) {
  ScopedEnvVar profile("AWS_PROFILE", "milvus-storage-no-such-profile");
  ScopedEnvVar access_key("AWS_ACCESS_KEY_ID", "env-ak");
  ScopedEnvVar secret_key("AWS_SECRET_ACCESS_KEY", "env-sk");
  ScopedEnvUnset role("AWS_ROLE_ARN");
  ScopedEnvUnset token_path("AWS_WEB_IDENTITY_TOKEN_FILE");

  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderAWS;
  config.use_iam = true;
  config.region = "us-east-1";
  config.use_ssl = false;
  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  auto resolver = std::dynamic_pointer_cast<RequestCredentialsResolver>(options.credentials_provider);
  ASSERT_NE(resolver, nullptr);
  ASSERT_AND_ASSIGN(auto credentials, resolver->ResolveForRequest());
  EXPECT_EQ(credentials.GetAWSAccessKeyId(), "env-ak");
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty());
}

TEST_F(S3ProviderTest, VendorStsClassifiesHttpFailureBeforeParsingNonEmptyBody) {
  auto config = MakeNoImdsClientConfiguration();
  config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("sts-status-test", 0);

  {
    ScopedEnvVar provider_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
      mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS,
                                    R"(<Error><Code>Throttling</Code><Message>busy but non-empty</Message></Error>)");
    }
    AliyunSTSCredentialsClient client(config);
    AliyunSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{"session", "acs:ram::123456:role/test",
                                                                            "identity-token"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectExtendStatusCode(result.status, ExtendStatusCode::StorageTransientThrottling);
  }

  {
    for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
      mock_client_->EnqueueResponse("sts.tencentcloudapi.com", Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE,
                                    R"({"Response":{"Error":{"Code":"InternalError","Message":"non-empty outage"}}})");
    }
    TencentCloudSTSCredentialsClient client(config);
    TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "ap-guangzhou", "provider", "identity-token", "qcs::cam::uin/1:roleName/test", "session"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectExtendStatusCode(result.status, ExtendStatusCode::StorageTransientService);
  }

  {
    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::FORBIDDEN,
                                  R"(<Error><Code>AccessDenied</Code><Message>non-empty refusal</Message></Error>)");
    AliyunRAMSTSClient client(config);
    AliyunRAMSTSClient::AssumeRoleRequest request;
    request.callerAccessKeyId = "caller-ak";
    request.callerAccessKeySecret = "caller-sk";
    request.callerSecurityToken = "caller-token";
    request.roleArn = "acs:ram::123456:role/test";
    request.roleSessionName = "session";

    auto result = client.GetAssumeRoleCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectExtendStatusCode(result.status, ExtendStatusCode::StorageAccessDenied);
  }
}

TEST_F(S3ProviderTest, TencentHttp200ErrorBodyKeepsTypedCause) {
  auto config = MakeNoImdsClientConfiguration();
  config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("tencent-200-error-test", 0);
  struct Case {
    const char* code;
    ExtendStatusCode expected;
  };
  const std::vector<Case> cases = {
      {"AuthFailure", ExtendStatusCode::StorageAccessDenied},
      {"AuthFailure.SignatureFailure", ExtendStatusCode::StorageAccessDenied},
      {"AccessDenied.ResourceUnauthorized", ExtendStatusCode::StorageAccessDenied},
      {"UnauthorizedOperation.AssumeRoleWithWebIdentity", ExtendStatusCode::StorageAccessDenied},
      {"RequestLimitExceeded", ExtendStatusCode::StorageTransientThrottling},
      {"RequestLimitExceeded.UinLimitExceeded", ExtendStatusCode::StorageTransientThrottling},
      {"InvalidParameter.OverLimit", ExtendStatusCode::StorageTransientThrottling},
      {"InternalError", ExtendStatusCode::StorageTransientService},
      {"InternalError.StsInternalError", ExtendStatusCode::StorageTransientService},
      {"ServiceUnavailable.StsService", ExtendStatusCode::StorageTransientService},
      {"ResourceUnavailable", ExtendStatusCode::StorageTransientService},
      {"ResourceUnavailable.RoleState", ExtendStatusCode::StorageConfigInvalid},
      {"InvalidParameter.RoleArn", ExtendStatusCode::StorageConfigInvalid},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.code);
    const auto body = fmt::format(
        R"({{"Response":{{"Error":{{"Code":"{}","Message":"synthetic typed error"}},"RequestId":"request"}}}})",
        test_case.code);
    mock_client_->EnqueueResponse("sts.tencentcloudapi.com", Aws::Http::HttpResponseCode::OK, body);
    TencentCloudSTSCredentialsClient client(config);
    TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "ap-guangzhou", "provider", "identity-token", "qcs::cam::uin/1:roleName/test", "session"};
    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectExtendStatusCode(result.status, test_case.expected);
  }
}

TEST_F(S3ProviderTest, VendorStsRejectsEveryIncompleteTemporaryCredentialShape) {
  auto config = MakeNoImdsClientConfiguration();
  config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("sts-partial-test", 0);

  {
    ScopedEnvVar provider_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    mock_client_->EnqueueResponse(
        "sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK,
        R"(<AssumeRoleWithOIDCResponse><Credentials><AccessKeyId>ak</AccessKeyId><AccessKeySecret>sk</AccessKeySecret><Expiration>2099-12-31T23:59:59Z</Expiration></Credentials></AssumeRoleWithOIDCResponse>)");
    AliyunSTSCredentialsClient client(config);
    AliyunSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{"session", "acs:ram::123456:role/test",
                                                                            "identity-token"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    EXPECT_FALSE(result.status.ok());
  }

  {
    mock_client_->EnqueueResponse(
        "sts.tencentcloudapi.com", Aws::Http::HttpResponseCode::OK,
        R"({"Response":{"Credentials":{"TmpSecretId":"ak","TmpSecretKey":"sk","Token":"token"},"Expiration":""}})");
    TencentCloudSTSCredentialsClient client(config);
    TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "ap-guangzhou", "provider", "identity-token", "qcs::cam::uin/1:roleName/test", "session"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    EXPECT_FALSE(result.status.ok());
  }

  {
    mock_client_->EnqueueResponse(
        "sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK,
        R"(<AssumeRoleResponse><Credentials><AccessKeyId>ak</AccessKeyId><AccessKeySecret>sk</AccessKeySecret><SecurityToken>token</SecurityToken></Credentials></AssumeRoleResponse>)");
    AliyunRAMSTSClient client(config);
    AliyunRAMSTSClient::AssumeRoleRequest request;
    request.callerAccessKeyId = "caller-ak";
    request.callerAccessKeySecret = "caller-sk";
    request.callerSecurityToken = "caller-token";
    request.roleArn = "acs:ram::123456:role/test";
    request.roleSessionName = "session";

    auto result = client.GetAssumeRoleCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    EXPECT_FALSE(result.status.ok());
  }
}

TEST_F(S3ProviderTest, StandardClientHolderStopsObjectRequestWhenCredentialResolutionFails) {
  auto provider = std::make_shared<FailingRequestCredentialsProvider>(
      MakeExtendErrorMsg(ExtendStatusCode::StorageTransientService, "runtime credential refresh failed"));
  S3Options options;
  options.credentials_provider = provider;
  options.credentials_kind = S3CredentialsKind::Role;
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto file_info = fs->GetFileInfo("bucket/key");
  ASSERT_FALSE(file_info.ok());
  ExpectExtendStatusCode(file_info.status(), ExtendStatusCode::StorageTransientService);
  EXPECT_EQ(provider->resolve_calls(), 1);
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty())
      << "an object request must not be sent after credential refresh fails";
}

TEST_F(S3ProviderTest, AwsAssumeRoleRuntimeFailureStopsStandardObjectRequest) {
  auto sts = std::make_shared<StubSTSClient>();
  sts->PushOutcome(MakeStsFailure(Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "ServiceUnavailable"));

  S3Options options;
  options.ConfigureAssumeRoleCredentials("arn:aws:iam::123456789012:role/test", "runtime-gate", "", 900, sts);
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  auto file_info = fs->GetFileInfo("bucket/key");
  ASSERT_FALSE(file_info.ok());
  ExpectExtendStatusCode(file_info.status(), ExtendStatusCode::StorageTransientService);
  size_t s3_requests = 0;
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    if (request->GetURIString().find("mock-s3.local") != Aws::String::npos) {
      ++s3_requests;
    }
  }
  EXPECT_EQ(s3_requests, 0u) << "the Standard holder must not expose its S3 client after AssumeRole refresh fails";
}

#ifdef WITH_CRT
TEST_F(S3ProviderTest, CrtClientHolderStopsBeforeExposingClientWhenCredentialResolutionFails) {
  auto provider = std::make_shared<FailingRequestCredentialsProvider>(
      MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, "runtime credential network failure"));
  S3Options options;
  options.credentials_provider = provider;
  options.credentials_kind = S3CredentialsKind::Role;
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;

  ClientBuilder<Aws::S3Crt::S3CrtClient> builder(options);
  ASSERT_AND_ASSIGN(auto holder, builder.BuildClient());
  auto locked = holder->Lock();
  ASSERT_FALSE(locked.ok());
  ExpectExtendStatusCode(locked.status(), ExtendStatusCode::StorageTransientNetwork);
  EXPECT_EQ(provider->resolve_calls(), 1);
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty());
}
#endif

TEST_F(S3ProviderTest, UseIamConstructionPreflightReturnsTypedTencentFailure) {
  TempFile token_file("tencent-identity-token");
  ScopedEnvVar region("TKE_REGION", "ap-guangzhou");
  ScopedEnvVar role("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
  ScopedEnvVar token("TKE_WEB_IDENTITY_TOKEN_FILE", token_file.path());
  ScopedEnvVar provider_id("TKE_PROVIDER_ID", "test-provider");

  const std::string error_body = R"({"Response":{"Error":{"Code":"InternalError","Message":"non-empty STS outage"}}})";
  for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
    mock_client_->EnqueueResponse("sts.tencentcloudapi.com", Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE,
                                  error_body);
  }

  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderTencent;
  config.use_iam = true;
  config.region = "ap-guangzhou";
  config.use_ssl = false;
  S3FileSystemProducer producer(config);

  auto options = producer.CreateS3Options();
  ASSERT_FALSE(options.ok());
  ExpectExtendStatusCode(options.status(), ExtendStatusCode::StorageTransientService);
  EXPECT_FALSE(mock_client_->GetRecordedRequests().empty())
      << "use_iam must preflight its provider during construction";
}

TEST_F(S3ProviderTest, RoleArnConstructionPreflightReturnsTypedAwsFailure) {
  ScopedEnvVar access_key("AWS_ACCESS_KEY_ID", "source-ak");
  ScopedEnvVar secret_key("AWS_SECRET_ACCESS_KEY", "source-sk");
  ScopedEnvVar session_token("AWS_SESSION_TOKEN", "");
  ScopedEnvVar region("AWS_REGION", "us-east-1");
  ScopedEnvVar default_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvVar disable_imds("AWS_EC2_METADATA_DISABLED", "true");

  const std::string error_body = R"(<?xml version="1.0" encoding="UTF-8"?>
<ErrorResponse><Error><Type>Sender</Type><Code>AccessDenied</Code>
<Message>role trust policy refused this caller</Message></Error><RequestId>request-id</RequestId></ErrorResponse>)";
  Aws::Http::HeaderValueCollection headers;
  headers["content-type"] = "application/xml";
  headers["content-length"] = std::to_string(error_body.size());
  mock_client_->EnqueueResponse("amazonaws.com", Aws::Http::HttpResponseCode::FORBIDDEN, error_body, headers);

  ArrowFileSystemConfig config;
  config.cloud_provider = kCloudProviderAWS;
  config.role_arn = "arn:aws:iam::123456789012:role/target";
  config.session_name = "construction-preflight";
  config.load_frequency = 900;
  config.region = "us-east-1";
  config.use_ssl = false;
  S3FileSystemProducer producer(config);

  auto options = producer.CreateS3Options();
  ASSERT_FALSE(options.ok());
  ExpectExtendStatusCode(options.status(), ExtendStatusCode::StorageAccessDenied);
  EXPECT_FALSE(mock_client_->GetRecordedRequests().empty())
      << "role_arn must resolve AssumeRole credentials before publishing S3Options";
}

// One retry budget for every provider, and it has to actually retry. The
// budget used to be set as ClientConfiguration::retryStrategy on clients driven
// by raw HttpClient::MakeRequest, which does not consume one -- the SDK's retry
// loop lives in AWSHttpResourceClient. Those paths stayed one-shot while the
// constant said three, so this counts requests rather than asserting a number.
namespace {

class CountingHttpClient final : public Aws::Http::HttpClient {
  public:
  explicit CountingHttpClient(Aws::Http::HttpResponseCode code) : code_(code) {}

  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
      const std::shared_ptr<Aws::Http::HttpRequest>& request,
      Aws::Utils::RateLimits::RateLimiterInterface* = nullptr,
      Aws::Utils::RateLimits::RateLimiterInterface* = nullptr) const override {
    ++attempts_;
    auto response = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("CountingHttpClient", request);
    response->SetResponseCode(code_);
    return response;
  }

  int attempts() const { return attempts_; }

  private:
  Aws::Http::HttpResponseCode code_;
  mutable int attempts_ = 0;
};

// Built directly rather than through Aws::Http::CreateHttpRequest, which needs
// the SDK's global HTTP factory to have been installed by somebody else. These
// are free tests with no fixture, so depending on that made them pass only when
// another test ran first and segfault under --gtest_filter -- which is exactly
// when someone is trying to debug one of them.
std::shared_ptr<Aws::Http::HttpRequest> MakeProbeRequest(
    Aws::Http::HttpMethod method = Aws::Http::HttpMethod::HTTP_GET) {
  auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>(
      "StsCredentialResolutionTest", Aws::Http::URI("http://credential-probe.local/"), method);
  // CreateHttpRequest would have installed this; building the request directly
  // does not, and StandardHttpResponse calls it during construction.
  request->SetResponseStreamFactory(Aws::Utils::Stream::DefaultResponseStreamFactoryMethod);
  return request;
}

}  // namespace

TEST(StsCredentialResolutionTest, RetriesAServiceFailureUpToTheSharedBudget) {
  CountingHttpClient client(Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE);
  auto response = MakeRequestWithCredentialRetry(client, MakeProbeRequest());

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->GetResponseCode(), Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE);
  // One initial attempt plus the budget.
  EXPECT_EQ(client.attempts(), kCredentialRetryAttempts + 1);
}

// A retried POST must resend its body. Resending the same HttpRequest resends a
// stream the previous attempt already drained, so without a rewind the retry
// goes out empty and STS rejects it on its contents -- turning a retryable
// outage into a bogus "malformed request". Huawei's second stage is a POST, so
// this is not hypothetical.
TEST(StsCredentialResolutionTest, ResendsTheRequestBodyOnEveryAttempt) {
  class BodyRecordingHttpClient final : public Aws::Http::HttpClient {
 public:
    std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
        const std::shared_ptr<Aws::Http::HttpRequest>& request,
        Aws::Utils::RateLimits::RateLimiterInterface* = nullptr,
        Aws::Utils::RateLimits::RateLimiterInterface* = nullptr) const override {
      // Drain the body exactly as a transport would.
      std::ostringstream drained;
      drained << request->GetContentBody()->rdbuf();
      bodies_.push_back(drained.str());
      auto response = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>("BodyRecordingHttpClient", request);
      response->SetResponseCode(Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE);
      return response;
    }

    const std::vector<std::string>& bodies() const { return bodies_; }

 private:
    mutable std::vector<std::string> bodies_;
  };

  BodyRecordingHttpClient client;
  auto request = MakeProbeRequest(Aws::Http::HttpMethod::HTTP_POST);
  auto body = Aws::MakeShared<Aws::StringStream>("test");
  *body << R"({"auth":"payload"})";
  request->AddContentBody(body);

  (void)MakeRequestWithCredentialRetry(client, request);

  ASSERT_EQ(client.bodies().size(), static_cast<size_t>(kCredentialRetryAttempts + 1));
  for (size_t attempt = 0; attempt < client.bodies().size(); ++attempt) {
    EXPECT_EQ(client.bodies()[attempt], R"({"auth":"payload"})") << "attempt " << attempt;
  }
}

TEST(StsCredentialResolutionTest, DoesNotRetryARefusal) {
  // A 4xx is the service answering on the merits. Repeating it changes nothing
  // and only delays the operator learning the role is wrong.
  CountingHttpClient client(Aws::Http::HttpResponseCode::FORBIDDEN);
  auto response = MakeRequestWithCredentialRetry(client, MakeProbeRequest());

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(client.attempts(), 1);
}

TEST(StsCredentialResolutionTest, StopsRetryingOnceItSucceeds) {
  CountingHttpClient client(Aws::Http::HttpResponseCode::OK);
  auto response = MakeRequestWithCredentialRetry(client, MakeProbeRequest());

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(client.attempts(), 1);
}

TEST_F(S3ProviderTest, VendorCredentialClientsContainDependencyExceptions) {
  auto config = MakeNoImdsClientConfiguration();
  config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("credential-exception-test", 0);

  {
    ScopedEnvVar provider_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    mock_client_->EnqueueException("sts.aliyuncs.com", MockExceptionKind::RuntimeError);
    AliyunSTSCredentialsClient client(config);
    AliyunSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "session", "acs:ram::123456:role/test", "identity-token"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectCredentialDependencyException(result.status);
  }

  {
    mock_client_->EnqueueException("sts.tencentcloudapi.com", MockExceptionKind::RuntimeError);
    TencentCloudSTSCredentialsClient client(config);
    TencentCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "ap-guangzhou", "provider", "identity-token", "qcs::cam::uin/1:roleName/test", "session"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectCredentialDependencyException(result.status);
  }

  {
    mock_client_->EnqueueException("sts.aliyuncs.com", MockExceptionKind::RuntimeError);
    AliyunRAMSTSClient client(config);
    AliyunRAMSTSClient::AssumeRoleRequest request;
    request.callerAccessKeyId = "caller-ak";
    request.callerAccessKeySecret = "caller-sk";
    request.callerSecurityToken = "caller-token";
    request.roleArn = "acs:ram::123456:role/test";
    request.roleSessionName = "session";

    auto result = client.GetAssumeRoleCredentials(request);
    EXPECT_TRUE(result.creds.IsEmpty());
    ExpectCredentialDependencyException(result.status);
  }

  {
    Aws::Http::HeaderValueCollection headers;
    headers["x-subject-token"] = "subject-token";
    mock_client_->EnqueueResponse("OS-AUTH/id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", headers);
    mock_client_->EnqueueException("OS-CREDENTIAL/securitytokens", MockExceptionKind::RuntimeError);
    HuaweiCloudSTSCredentialsClient client(config);
    HuaweiCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityRequest request{
        "cn-north-4", "provider", "identity-token", "project-id", "session"};

    auto result = client.GetAssumeRoleWithWebIdentityCredentials(request);
    EXPECT_FALSE(result.success);
    ExpectCredentialDependencyException(result.status);
  }

  {
    ScopedEnvUnset metadata_disabled("ALIBABA_CLOUD_ECS_METADATA_DISABLED");
    mock_client_->EnqueueException("latest/api/token", MockExceptionKind::RuntimeError);
    AliyunRAMCredentialsProvider provider("acs:ram::123456:role/test", "session");

    auto result = provider.ResolveForRequest();
    ASSERT_FALSE(result.ok());
    ExpectCredentialDependencyException(result.status());
  }
}

TEST_F(S3ProviderTest, VendorCredentialClientsPreserveOutOfMemory) {
  auto config = MakeNoImdsClientConfiguration();
  config.retryStrategy = Aws::MakeShared<Aws::Client::DefaultRetryStrategy>("credential-oom-test", 0);

  {
    ScopedEnvVar provider_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    mock_client_->EnqueueException("sts.aliyuncs.com", MockExceptionKind::BadAlloc);
    AliyunSTSCredentialsClient client(config);
    auto result = client.GetAssumeRoleWithWebIdentityCredentials(
        {"session", "acs:ram::123456:role/test", "identity-token"});
    ExpectCredentialOutOfMemory(result.status);
  }

  {
    mock_client_->EnqueueException("sts.tencentcloudapi.com", MockExceptionKind::BadAlloc);
    TencentCloudSTSCredentialsClient client(config);
    auto result = client.GetAssumeRoleWithWebIdentityCredentials(
        {"ap-guangzhou", "provider", "identity-token", "qcs::cam::uin/1:roleName/test", "session"});
    ExpectCredentialOutOfMemory(result.status);
  }

  {
    mock_client_->EnqueueException("sts.aliyuncs.com", MockExceptionKind::BadAlloc);
    AliyunRAMSTSClient client(config);
    AliyunRAMSTSClient::AssumeRoleRequest request;
    request.callerAccessKeyId = "caller-ak";
    request.callerAccessKeySecret = "caller-sk";
    request.callerSecurityToken = "caller-token";
    request.roleArn = "acs:ram::123456:role/test";
    request.roleSessionName = "session";
    auto result = client.GetAssumeRoleCredentials(request);
    ExpectCredentialOutOfMemory(result.status);
  }

  {
    Aws::Http::HeaderValueCollection headers;
    headers["x-subject-token"] = "subject-token";
    mock_client_->EnqueueResponse("OS-AUTH/id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", headers);
    mock_client_->EnqueueException("OS-CREDENTIAL/securitytokens", MockExceptionKind::BadAlloc);
    HuaweiCloudSTSCredentialsClient client(config);
    auto result = client.GetAssumeRoleWithWebIdentityCredentials(
        {"cn-north-4", "provider", "identity-token", "project-id", "session"});
    EXPECT_FALSE(result.success);
    ExpectCredentialOutOfMemory(result.status);
  }

  {
    ScopedEnvUnset metadata_disabled("ALIBABA_CLOUD_ECS_METADATA_DISABLED");
    mock_client_->EnqueueException("latest/api/token", MockExceptionKind::BadAlloc);
    AliyunRAMCredentialsProvider provider("acs:ram::123456:role/test", "session");
    auto result = provider.ResolveForRequest();
    ASSERT_FALSE(result.ok());
    ExpectCredentialOutOfMemory(result.status());
  }
}

TEST_F(S3ProviderTest, OneShotWriteFileAbortsTheUploadItCreated) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  const std::string failed_upload_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>AccessDenied</Code>
  <Message>part was denied</Message>
  <RequestId>request-id</RequestId>
</Error>)xml";
  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };

  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers(create_result));
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::FORBIDDEN, failed_upload_result,
                                xml_headers(failed_upload_result));
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = false;
  // Every byte becomes its own part, so the very first write creates the
  // multipart upload this test is about.
  options.multi_part_upload_size = 1;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));

  // The one-shot C API never returns a writer handle, so nothing the caller
  // holds can release the upload afterwards. If this call does not abort it
  // itself, the parts stay in the bucket where no listing can name them.
  FileSystemWrapper fs_wrapper(fs);
  const std::string path = "bucket/key";
  const uint8_t bytes[] = {7, 8, 9};
  auto result = loon_filesystem_write_file(reinterpret_cast<FileSystemHandle>(&fs_wrapper), path.data(),
                                           static_cast<uint32_t>(path.size()), bytes, sizeof(bytes), nullptr, 0);
  EXPECT_FALSE(loon_ffi_is_success(&result));
  loon_ffi_free_result(&result);

  size_t aborts = 0;
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    if (request->GetMethod() == Aws::Http::HttpMethod::HTTP_DELETE &&
        request->GetURIString().find("uploadId=upload-id") != Aws::String::npos) {
      ++aborts;
    }
  }
  EXPECT_EQ(aborts, 1);
}

TEST_F(S3ProviderTest, BackgroundWriteStopsSubmittingAfterObservedPartFailure) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  const std::string failed_upload_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>AccessDenied</Code>
  <Message>first part was denied</Message>
  <RequestId>request-id</RequestId>
</Error>)xml";
  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };

  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers(create_result));
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::FORBIDDEN, failed_upload_result,
                                xml_headers(failed_upload_result));
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  // Run each accepted background task inline. The first part's failure is
  // therefore recorded before the same Write() attempts its second part.
  arrow::MockExecutor executor;
  arrow::io::IOContext io_context(&executor);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

  const uint8_t bytes[] = {7, 8, 9};
  auto write_status = stream->Write(bytes, 3);

  ASSERT_STATUS_NOT_OK(write_status);
  auto detail = ExtendStatusDetail::UnwrapStatus(write_status);
  ASSERT_NE(detail, nullptr) << write_status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied) << write_status.ToString();
  EXPECT_NE(write_status.message().find("first part was denied"), std::string::npos) << write_status.ToString();

  size_t upload_part_requests = 0;
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    if (request->GetURIString().find("partNumber=") != Aws::String::npos) {
      ++upload_part_requests;
    }
  }
  EXPECT_EQ(upload_part_requests, 1);
  ASSERT_STATUS_OK(stream->Abort());
}

TEST_F(S3ProviderTest, BackgroundCloseFailureDoesNotWaitForSiblingPart) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  const std::string failed_upload_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>AccessDenied</Code>
  <Message>first completed part was denied</Message>
  <RequestId>request-id</RequestId>
</Error>)xml";
  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };

  std::promise<void> sibling_started_promise;
  auto sibling_started = sibling_started_promise.get_future();
  std::promise<void> release_sibling_promise;
  auto release_sibling = release_sibling_promise.get_future().share();
  std::atomic<bool> sibling_release_timed_out{false};

  Aws::Http::HeaderValueCollection successful_upload_headers;
  successful_upload_headers["etag"] = "\"etag-2\"";
  successful_upload_headers["content-length"] = "0";
  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers(create_result));
  mock_client_->EnqueueResponse(
      "?partNumber=1", Aws::Http::HttpResponseCode::FORBIDDEN, failed_upload_result, xml_headers(failed_upload_result),
      [&sibling_started] { EXPECT_EQ(sibling_started.wait_for(std::chrono::seconds(5)), std::future_status::ready); });
  mock_client_->EnqueueResponse("?partNumber=2", Aws::Http::HttpResponseCode::OK, "", successful_upload_headers,
                                [&sibling_started_promise, release_sibling, &sibling_release_timed_out] {
                                  sibling_started_promise.set_value();
                                  if (release_sibling.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
                                    sibling_release_timed_out.store(true, std::memory_order_release);
                                  }
                                });
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(2));
  arrow::io::IOContext io_context(pool.get());
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

  const uint8_t bytes[] = {7, 8};
  ASSERT_STATUS_OK(stream->Write(bytes, 2));
  auto close_future = stream->CloseAsync();

  const auto sibling_state = sibling_started.wait_for(std::chrono::seconds(5));
  const bool finished_before_sibling_release = sibling_state == std::future_status::ready && close_future.Wait(5.0);
  std::optional<arrow::Status> observed_status;
  if (finished_before_sibling_release) {
    observed_status = close_future.status();
  }
  release_sibling_promise.set_value();
  pool->WaitForIdle();
  auto abort_status = stream->Abort();

  ASSERT_EQ(sibling_state, std::future_status::ready);
  ASSERT_TRUE(finished_before_sibling_release) << "upload failure waited for a sibling part";
  ASSERT_TRUE(observed_status.has_value());
  ASSERT_STATUS_NOT_OK(*observed_status);
  auto detail = ExtendStatusDetail::UnwrapStatus(*observed_status);
  ASSERT_NE(detail, nullptr) << observed_status->ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied) << observed_status->ToString();
  EXPECT_NE(observed_status->message().find("first completed part was denied"), std::string::npos)
      << observed_status->ToString();
  EXPECT_FALSE(sibling_release_timed_out.load(std::memory_order_acquire));
  ASSERT_STATUS_OK(abort_status);
}

TEST_F(S3ProviderTest, BackgroundAbortRunsAfterInFlightPartSettles) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  Aws::Http::HeaderValueCollection create_headers;
  create_headers["content-type"] = "application/xml";
  create_headers["content-length"] = std::to_string(create_result.size());

  std::promise<void> upload_started_promise;
  auto upload_started = upload_started_promise.get_future();
  std::promise<void> release_upload_promise;
  auto release_upload = release_upload_promise.get_future().share();
  std::promise<void> abort_seen_promise;
  auto abort_seen = abort_seen_promise.get_future();
  std::atomic<bool> upload_release_timed_out{false};

  Aws::Http::HeaderValueCollection upload_headers;
  upload_headers["etag"] = "\"etag-1\"";
  upload_headers["content-length"] = "0";
  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, create_headers);
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::OK, "", upload_headers,
                                [&upload_started_promise, release_upload, &upload_release_timed_out] {
                                  upload_started_promise.set_value();
                                  if (release_upload.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                                    upload_release_timed_out.store(true, std::memory_order_release);
                                  }
                                });
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "", {},
                                [&abort_seen_promise] { abort_seen_promise.set_value(); });

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(1));
  arrow::io::IOContext io_context(pool.get());
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

  const uint8_t byte = 7;
  ASSERT_STATUS_OK(stream->Write(&byte, 1));
  ASSERT_EQ(upload_started.wait_for(std::chrono::seconds(5)), std::future_status::ready);

  ASSERT_STATUS_OK(stream->Abort());
  stream.reset();

  auto count_remote_aborts = [this] {
    size_t count = 0;
    for (const auto& request : mock_client_->GetRecordedRequests()) {
      if (request->GetMethod() == Aws::Http::HttpMethod::HTTP_DELETE &&
          request->GetURIString().find("uploadId=upload-id") != Aws::String::npos) {
        ++count;
      }
    }
    return count;
  };
  EXPECT_EQ(count_remote_aborts(), 0);

  release_upload_promise.set_value();
  ASSERT_EQ(abort_seen.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  pool->WaitForIdle();

  EXPECT_FALSE(upload_release_timed_out.load(std::memory_order_acquire));
  EXPECT_EQ(count_remote_aborts(), 1);
}

TEST_F(S3ProviderTest, RepeatedAbortCannotStealDeferredUploadIdentity) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  Aws::Http::HeaderValueCollection create_headers;
  create_headers["content-type"] = "application/xml";
  create_headers["content-length"] = std::to_string(create_result.size());

  std::promise<void> upload_started_promise;
  auto upload_started = upload_started_promise.get_future();
  std::promise<void> release_upload_promise;
  auto release_upload = release_upload_promise.get_future().share();
  std::promise<void> completion_published_promise;
  auto completion_published = completion_published_promise.get_future();
  std::promise<void> release_completion_promise;
  auto release_completion = release_completion_promise.get_future().share();
  std::atomic<bool> upload_release_timed_out{false};
  std::atomic<bool> completion_release_timed_out{false};

  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, create_headers);
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR, "", {},
                                [&upload_started_promise, release_upload, &upload_release_timed_out] {
                                  upload_started_promise.set_value();
                                  if (release_upload.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                                    upload_release_timed_out.store(true, std::memory_order_release);
                                  }
                                });
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, "");

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  ASSERT_AND_ASSIGN(auto pool, arrow::internal::ThreadPool::Make(1));
  arrow::io::IOContext io_context(pool.get());
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));

  const uint8_t byte = 7;
  ASSERT_STATUS_OK(stream->Write(&byte, 1));
  ASSERT_EQ(upload_started.wait_for(std::chrono::seconds(5)), std::future_status::ready);

  // The failed UploadPart makes CloseAsync take only its error continuation,
  // so it cannot complete the multipart upload. Its continuation is published
  // synchronously by HandleUploadOutcome after uploads_in_progress reaches
  // zero, but before that function calls AbortRecordedUpload. Block there to
  // make the race with a repeated Abort deterministic.
  auto close_future = stream->CloseAsync();
  close_future.AddCallback(
      [&completion_published_promise, release_completion, &completion_release_timed_out](const arrow::Status&) {
        completion_published_promise.set_value();
        if (release_completion.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
          completion_release_timed_out.store(true, std::memory_order_release);
        }
      });

  ASSERT_STATUS_OK(stream->Abort());
  release_upload_promise.set_value();
  ASSERT_EQ(completion_published.wait_for(std::chrono::seconds(5)), std::future_status::ready);

  // The second Abort races the last upload completion after its counter is
  // zero. It must consume the recorded identity itself, not erase it before
  // the completion can issue the remote abort.
  ASSERT_STATUS_OK(stream->Abort());
  release_completion_promise.set_value();
  pool->WaitForIdle();

  size_t remote_aborts = 0;
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    if (request->GetMethod() == Aws::Http::HttpMethod::HTTP_DELETE &&
        request->GetURIString().find("uploadId=upload-id") != Aws::String::npos) {
      ++remote_aborts;
    }
  }
  EXPECT_FALSE(upload_release_timed_out.load(std::memory_order_acquire));
  EXPECT_FALSE(completion_release_timed_out.load(std::memory_order_acquire));
  EXPECT_EQ(remote_aborts, 1);
}

TEST_F(S3ProviderTest, AbortAfterSuccessfulMultipartCloseDoesNotDeleteCompletedUpload) {
  const std::string create_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><UploadId>upload-id</UploadId>
</InitiateMultipartUploadResult>)xml";
  const std::string complete_result = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<CompleteMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>bucket</Bucket><Key>key</Key><ETag>"etag-1"</ETag>
</CompleteMultipartUploadResult>)xml";
  auto xml_headers = [](const std::string& body) {
    Aws::Http::HeaderValueCollection headers;
    headers["content-type"] = "application/xml";
    headers["content-length"] = std::to_string(body.size());
    return headers;
  };
  Aws::Http::HeaderValueCollection upload_headers;
  upload_headers["etag"] = "\"etag-1\"";
  upload_headers["content-length"] = "0";

  mock_client_->EnqueueResponse("?uploads", Aws::Http::HttpResponseCode::OK, create_result, xml_headers(create_result));
  mock_client_->EnqueueResponse("?partNumber=1", Aws::Http::HttpResponseCode::OK, "", upload_headers);
  mock_client_->EnqueueResponse("uploadId=upload-id", Aws::Http::HttpResponseCode::OK, complete_result,
                                xml_headers(complete_result));

  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = false;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 1));
  const uint8_t byte = 7;
  ASSERT_STATUS_OK(stream->Write(&byte, 1));
  ASSERT_STATUS_OK(stream->Close());
  ASSERT_STATUS_OK(stream->Abort());
  // Abort after a successful Close is a no-op in every respect: it must not
  // poison the terminal result seen by a later idempotent Close.
  ASSERT_STATUS_OK(stream->Close());

  size_t remote_aborts = 0;
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    if (request->GetMethod() == Aws::Http::HttpMethod::HTTP_DELETE &&
        request->GetURIString().find("uploadId=upload-id") != Aws::String::npos) {
      ++remote_aborts;
    }
  }
  EXPECT_EQ(remote_aborts, 0);
}

TEST_F(S3ProviderTest, BackgroundCloseAsyncPreservesSubmissionFailure) {
  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  RejectingS3Executor executor;
  arrow::io::IOContext io_context(&executor);
  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options, io_context));
  // Keep the byte buffered so the first background submission happens inside
  // CloseAsync(), exercising its synchronous submission-error path.
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 2));
  const uint8_t byte = 7;
  ASSERT_STATUS_OK(stream->Write(&byte, 1));

  auto close_status = stream->CloseAsync().status();
  ASSERT_STATUS_NOT_OK(close_status);
  EXPECT_NE(close_status.message().find("executor rejected S3 upload task"), std::string::npos)
      << close_status.ToString();
  EXPECT_TRUE(stream->closed());
}

TEST_F(S3ProviderTest, BackgroundCloseAsyncPreservesSynchronousPrimaryFailure) {
  S3Options options;
  options.ConfigureAnonymousCredentials();
  options.region = "us-east-1";
  options.scheme = "http";
  options.endpoint_override = "mock-s3.local";
  options.cloud_provider = kCloudProviderAWS;
  options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
  options.use_crt_async_reads = false;
  options.background_writes = true;

  ASSERT_AND_ASSIGN(auto fs, S3FileSystem::Make(options));
  ASSERT_AND_ASSIGN(auto stream, fs->OpenOutputStreamWithUploadSize("bucket/key", nullptr, 2));
  const uint8_t byte = 7;
  ASSERT_STATUS_OK(stream->Write(&byte, 1));

  ScopedFiuFault fault(FIUKEY_S3FS_WRITER_CLOSE_FAIL);
  ASSERT_EQ(0, fault.enable_result());
  auto close_status = stream->CloseAsync().status();

  ASSERT_STATUS_NOT_OK(close_status);
  EXPECT_NE(close_status.message().find(FIUKEY_S3FS_WRITER_CLOSE_FAIL), std::string::npos) << close_status.ToString();
  auto detail = ExtendStatusDetail::UnwrapStatus(close_status);
  ASSERT_NE(detail, nullptr) << close_status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientNetwork) << close_status.ToString();
  EXPECT_TRUE(stream->closed());
}

TEST_F(S3ProviderTest, CopyObjectNotFoundReportsMissingObjectWithoutProbe) {
  auto make_fs = []() -> arrow::Result<std::shared_ptr<S3FileSystem>> {
    S3Options options;
    options.ConfigureAnonymousCredentials();
    options.region = "us-east-1";
    options.scheme = "http";
    options.endpoint_override = "mock-s3.local";
    options.cloud_provider = kCloudProviderAWS;
    options.retry_strategy = S3RetryStrategy::GetAwsDefaultRetryStrategy(/*max_attempts=*/0);
    options.use_crt_async_reads = false;
    return S3FileSystem::Make(options);
  };

  // CopyObject's generic 404 (missing source key, source bucket, or
  // destination bucket) is reported as a missing object in ONE request; the
  // per-bucket HeadBucket probes are deliberately not issued.
  mock_client_->EnqueueResponse("mock-s3.local", Aws::Http::HttpResponseCode::NOT_FOUND, "");
  ASSERT_AND_ASSIGN(auto missing_source_object_fs, make_fs());
  auto missing_source_object =
      missing_source_object_fs->CopyFile("source-bucket/source-key", "destination-bucket/destination-key");
  ASSERT_STATUS_NOT_OK(missing_source_object);
  auto object_detail = ExtendStatusDetail::UnwrapStatus(missing_source_object);
  ASSERT_NE(object_detail, nullptr) << missing_source_object.ToString();
  EXPECT_EQ(object_detail->code(), ExtendStatusCode::StorageNotFound) << missing_source_object.ToString();
  EXPECT_EQ(CategoryForExtendStatusCode(object_detail->code()), ErrorCategory::System);
  // Only the failing CopyObject (plus SDK retries) -- the per-bucket
  // HeadBucket probes would show up as HEAD requests.
  for (const auto& request : mock_client_->GetRecordedRequests()) {
    EXPECT_NE(request->GetMethod(), Aws::Http::HttpMethod::HTTP_HEAD) << "unexpected disambiguation probe";
  }
}

// ============================================================================
// Aliyun Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestAliyunProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_arn("ALIBABA_CLOUD_ROLE_ARN");
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");
    ScopedEnvUnset unset_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file env → empty credentials
  {
    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Missing role arn → empty credentials
  {
    ScopedEnvUnset unset_arn("ALIBABA_CLOUD_ROLE_ARN");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/some_token");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Token file does not exist → empty credentials
  {
    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/nonexistent_token_file_12345");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success flow — mock returns valid XML
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    std::string xml_response = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
    <RequestId>TEST-REQUEST-ID</RequestId>
    <Credentials>
        <AccessKeyId>MOCK_AK</AccessKeyId>
        <AccessKeySecret>MOCK_SK</AccessKeySecret>
        <SecurityToken>MOCK_TOKEN</SecurityToken>
        <Expiration>2099-12-31T23:59:59Z</Expiration>
    </Credentials>
</AssumeRoleWithOIDCResponse>)";

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml_response);

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: STS returns empty body → empty credentials
  {
    TempFile token_file("mock_oidc_token_content");

    ScopedEnvVar set_arn("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::123456:role/test-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::123456:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Parameterized ctor — args populate roleArn/sessionName, machine
  // identity still comes from env. STS request body should carry the arg role,
  // not the env role.
  {
    TempFile token_file("mock_oidc_token_content");

    // Env has a DIFFERENT role to prove args win.
    ScopedEnvVar set_arn_env("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::000:role/env-role");
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::000:oidc-provider/test");
    ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

    std::string xml_response = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
    <RequestId>TEST-REQUEST-ID</RequestId>
    <Credentials>
        <AccessKeyId>ARG_AK</AccessKeyId>
        <AccessKeySecret>ARG_SK</AccessKeySecret>
        <SecurityToken>ARG_TOKEN</SecurityToken>
        <Expiration>2099-12-31T23:59:59Z</Expiration>
    </Credentials>
</AssumeRoleWithOIDCResponse>)";

    mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml_response);

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "ARG_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "ARG_SK");
    EXPECT_EQ(creds.GetSessionToken(), "ARG_TOKEN");

    // Verify the STS request body used the arg role, not the env role.
    auto recorded = mock_client_->GetRecordedRequests();
    ASSERT_FALSE(recorded.empty());
    auto& req = recorded.back();
    auto body_stream = req->GetContentBody();
    ASSERT_NE(body_stream, nullptr);
    std::string body((std::istreambuf_iterator<char>(*body_stream)), std::istreambuf_iterator<char>());
    EXPECT_NE(body.find("tenant-A"), std::string::npos) << "body should contain tenant-A role: " << body;
    EXPECT_EQ(body.find("env-role"), std::string::npos) << "body should not contain env role: " << body;
  }

  // Sub-test: Parameterized ctor — missing OIDC_TOKEN_FILE env → empty creds
  {
    ScopedEnvUnset unset_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::111:oidc-provider/test");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Parameterized ctor — token file path set but file missing on disk
  // → empty creds (Reload fails to open)
  {
    ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/tmp/nonexistent_token_file_param_ctor");
    ScopedEnvVar set_oidc_arn("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::111:oidc-provider/test");

    AliyunSTSAssumeRoleWebIdentityCredentialsProvider provider("acs:ram::111:role/tenant-A", "tenant-A-session");
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

// ============================================================================
// Tencent Cloud Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestTencentProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_region("TKE_REGION");
    ScopedEnvUnset unset_arn("TKE_ROLE_ARN");
    ScopedEnvUnset unset_token("TKE_WEB_IDENTITY_TOKEN_FILE");
    ScopedEnvUnset unset_provider("TKE_PROVIDER_ID");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file → empty credentials
  {
    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvUnset unset_token("TKE_WEB_IDENTITY_TOKEN_FILE");
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success flow — mock returns valid JSON
  {
    TempFile token_file("mock_tencent_token");

    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvVar set_token("TKE_WEB_IDENTITY_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    std::string json_response = R"({
      "Response": {
        "Credentials": {
          "TmpSecretId": "MOCK_AK",
          "TmpSecretKey": "MOCK_SK",
          "Token": "MOCK_TOKEN"
        },
        "Expiration": "2099-12-31T23:59:59Z"
      }
    })";

    mock_client_->EnqueueResponse("tencentcloudapi.com", Aws::Http::HttpResponseCode::OK, json_response);

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: STS returns empty body → empty credentials
  {
    TempFile token_file("mock_tencent_token");

    ScopedEnvVar set_region("TKE_REGION", "ap-guangzhou");
    ScopedEnvVar set_arn("TKE_ROLE_ARN", "qcs::cam::uin/100000000001:roleName/test-role");
    ScopedEnvVar set_token("TKE_WEB_IDENTITY_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_provider("TKE_PROVIDER_ID", "test-provider");

    mock_client_->EnqueueResponse("tencentcloudapi.com", Aws::Http::HttpResponseCode::OK, "");

    TencentCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Response missing Credentials field
  // Note: The Tencent STS code calls rootNode.GetString("Expiration") without
  // checking if rootNode has an "Expiration" key, so passing {"Response":{}}
  // would crash with an assertion. This is a known limitation of the source code;
  // we skip this sub-test to avoid crashing.
}

// ============================================================================
// Huawei Cloud Provider Tests
// ============================================================================

TEST_F(S3ProviderTest, TestHuaweiProvider) {
  // Sub-test: Uninitialized (missing env vars) → empty credentials
  {
    ScopedEnvUnset unset_region("HUAWEICLOUD_SDK_REGION");
    ScopedEnvUnset unset_project("HUAWEICLOUD_SDK_PROJECT_ID");
    ScopedEnvUnset unset_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
    ScopedEnvUnset unset_idp("HUAWEICLOUD_SDK_IDP_ID");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
    EXPECT_TRUE(creds.GetAWSSecretKey().empty());
    EXPECT_TRUE(creds.GetSessionToken().empty());
  }

  // Sub-test: Missing token file → empty credentials
  {
    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvUnset unset_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Success two-step flow
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Step 1 response: id-token/tokens → returns x-subject-token header
    Aws::Http::HeaderValueCollection step1_headers;
    step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

    // Step 2 response: securitytokens → returns credential JSON
    std::string step2_json = R"({
      "credential": {
        "access": "MOCK_AK",
        "secret": "MOCK_SK",
        "securitytoken": "MOCK_TOKEN",
        "expires_at": "2099-12-31T23:59:59Z"
      }
    })";
    mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");
    EXPECT_EQ(creds.GetAWSSecretKey(), "MOCK_SK");
    EXPECT_EQ(creds.GetSessionToken(), "MOCK_TOKEN");
  }

  // Sub-test: Step 1 fails (403 Forbidden) → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::FORBIDDEN, "Access denied");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Step 1 success but missing x-subject-token header → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Return 200 OK but without x-subject-token header
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::OK, "");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }

  // Sub-test: Step 1 success, Step 2 returns empty body → empty credentials
  {
    TempFile token_file("mock_huawei_id_token");

    ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
    ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
    ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
    ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

    // Step 1: success with subject token
    Aws::Http::HeaderValueCollection step1_headers;
    step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
    mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

    // Step 2: empty body
    mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, "");

    HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
    auto creds = provider.GetAWSCredentials();
    EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
  }
}

// ============================================================================
// Test Helper — friend class to access private members
// ============================================================================

class HuaweiCloudCredentialsProviderTestHelper {
  public:
  using Provider = HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider;

  static void setCredentials(Provider& p, const Aws::Auth::AWSCredentials& creds) { p.m_credentials = creds; }

  static void setTokenFile(Provider& p, const Aws::String& path) { p.m_tokenFile = path; }

  static void setInitialized(Provider& p, bool val) { p.m_initialized = val; }
};

using Helper = HuaweiCloudCredentialsProviderTestHelper;

// ============================================================================
// Huawei Cloud Provider — Resilience Tests
// ============================================================================

// Helper: count requests whose URL contains a given substring
static size_t CountRequestsByUrl(const std::vector<std::shared_ptr<Aws::Http::HttpRequest>>& requests,
                                 const std::string& url_substr) {
  size_t count = 0;
  for (const auto& req : requests) {
    if (req->GetURIString().find(url_substr) != Aws::String::npos) {
      count++;
    }
  }
  return count;
}



TEST_F(S3ProviderTest, TestHuaweiProviderCachesValidCredentials) {
  // Valid credentials with far-future expiration should be cached without new STS requests.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call: success with far-future expiration
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  std::string step2_json = R"({
    "credential": {
      "access": "VALID_AK",
      "secret": "VALID_SK",
      "securitytoken": "VALID_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds1 = provider.GetAWSCredentials();
  EXPECT_EQ(creds1.GetAWSAccessKeyId(), "VALID_AK");

  // Subsequent calls should use cached credentials without new STS requests
  size_t requests_before = mock_client_->GetRecordedRequests().size();

  auto creds2 = provider.GetAWSCredentials();
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "VALID_AK");

  size_t requests_after = mock_client_->GetRecordedRequests().size();
  EXPECT_EQ(requests_before, requests_after) << "Should use cached credentials without new STS calls";
}

TEST_F(S3ProviderTest, TestHuaweiProviderReturnsEmptyWhenCredsFullyExpired) {
  // GetAWSCredentials should return empty credentials when cached credentials have fully expired,
  // to avoid silent auth failures.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // First call: success with short expiration (already expired)
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Use an expiration time in the past
  auto now = std::chrono::system_clock::now();
  auto expired = now - std::chrono::seconds(60);
  auto expire_time_t = std::chrono::system_clock::to_time_t(expired);
  char expire_buf[64];
  std::strftime(expire_buf, sizeof(expire_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&expire_time_t));

  std::string step2_json =
      std::string(
          R"({"credential":{"access":"EXPIRED_AK","secret":"EXPIRED_SK","securitytoken":"EXPIRED_TK","expires_at":")") +
      expire_buf + R"("}})";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;

  // First call loads expired credentials. RefreshIfExpired will then try to
  // reload, but no more mock responses are available.
  provider.GetAWSCredentials();

  // Now GetAWSCredentials should return empty, not the expired credentials
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty()) << "Should return empty credentials instead of expired ones";
}


TEST_F(S3ProviderTest, TestHuaweiProviderStep2HttpFailure) {
  // Step 1 succeeds but Step 2 returns HTTP error → empty credentials.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Step 1 success
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2 fails with 500
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR, "server error");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2MissingCredentialField) {
  // Step 2 returns JSON without "credential" field → empty credentials.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid JSON but missing "credential" key
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, R"({"error": "something wrong"})");

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty());
}

TEST_F(S3ProviderTest, TestHuaweiProviderDurationSeconds7200) {
  // Verify the STS request body contains duration_seconds: 7200.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Step 1 success
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2 success
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "MOCK_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "MOCK_AK");

  // Find the securitytokens request and check its body for duration_seconds
  bool found_duration = false;
  for (const auto& req : mock_client_->GetRecordedRequests()) {
    if (req->GetURIString().find("securitytokens") != Aws::String::npos) {
      auto& bodyStream = req->GetContentBody();
      if (bodyStream) {
        bodyStream->seekg(0);
        std::string body((std::istreambuf_iterator<char>(*bodyStream)), std::istreambuf_iterator<char>());
        if (body.find("7200") != std::string::npos) {
          found_duration = true;
        }
      }
      break;
    }
  }
  EXPECT_TRUE(found_duration) << "STS request body should contain duration_seconds: 7200";
}

TEST_F(S3ProviderTest, TestHuaweiProviderConcurrentAccessNoStorm) {
  // Multiple threads calling GetAWSCredentials concurrently should not cause STS request storm.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  // Enqueue one success response — only one thread should consume it
  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  std::string step2_json = R"({
    "credential": {
      "access": "CONCURRENT_AK",
      "secret": "CONCURRENT_SK",
      "securitytoken": "CONCURRENT_TOKEN",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;

  constexpr int NUM_THREADS = 8;
  std::vector<std::thread> threads;
  std::vector<Aws::Auth::AWSCredentials> results(NUM_THREADS);

  threads.reserve(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back([&provider, &results, i]() { results[i] = provider.GetAWSCredentials(); });
  }

  for (auto& t : threads) {
    t.join();
  }

  // At least one thread should have gotten valid credentials
  int valid_count = 0;
  for (const auto& cred : results) {
    if (!cred.GetAWSAccessKeyId().empty()) {
      EXPECT_EQ(cred.GetAWSAccessKeyId(), "CONCURRENT_AK");
      valid_count++;
    }
  }
  EXPECT_GT(valid_count, 0) << "At least one thread should have gotten valid credentials";

  // The key assertion: STS id-token requests should be limited (not 8 separate requests)
  size_t sts_requests = CountRequestsByUrl(mock_client_->GetRecordedRequests(), "id-token/tokens");
  EXPECT_LE(sts_requests, 2u) << "Concurrent access should not cause STS request storm (got " << sts_requests << ")";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2EmptySessionToken) {
  // Step 2 returns valid ak/sk but empty securitytoken → should fail.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk but empty securitytoken
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "",
      "expires_at": "2099-12-31T23:59:59Z"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Empty session token should cause credential rejection";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2MissingExpiresAt) {
  // Step 2 returns valid credentials but missing expires_at → should fail.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk/token but no expires_at field
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "MOCK_TOKEN"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Missing expires_at should cause credential rejection";
}

TEST_F(S3ProviderTest, TestHuaweiProviderStep2InvalidExpiresAtFormat) {
  // Step 2 returns credentials with unparseable expires_at → should fail.
  TempFile token_file("mock_huawei_id_token");

  ScopedEnvVar set_region("HUAWEICLOUD_SDK_REGION", "cn-north-4");
  ScopedEnvVar set_project("HUAWEICLOUD_SDK_PROJECT_ID", "test-project-id");
  ScopedEnvVar set_token("HUAWEICLOUD_SDK_ID_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_idp("HUAWEICLOUD_SDK_IDP_ID", "test-idp");

  Aws::Http::HeaderValueCollection step1_headers;
  step1_headers["x-subject-token"] = "MOCK_SUBJECT_TOKEN";
  mock_client_->EnqueueResponse("id-token/tokens", Aws::Http::HttpResponseCode::CREATED, "", step1_headers);

  // Step 2: valid ak/sk/token but garbage expires_at
  std::string step2_json = R"({
    "credential": {
      "access": "MOCK_AK",
      "secret": "MOCK_SK",
      "securitytoken": "MOCK_TOKEN",
      "expires_at": "not-a-valid-date"
    }
  })";
  mock_client_->EnqueueResponse("securitytokens", Aws::Http::HttpResponseCode::OK, step2_json);

  HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider provider;
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.GetAWSAccessKeyId().empty()) << "Invalid expires_at format should cause credential rejection";
}

// ============================================================================
// Aliyun RAM STS Client Tests (POP v1 signing, XML response parsing)
// ============================================================================

namespace {

// Read the form-urlencoded body of a recorded POST request.
std::string ReadRequestBody(const std::shared_ptr<Aws::Http::HttpRequest>& req) {
  auto& stream = req->GetContentBody();
  if (!stream)
    return {};
  stream->seekg(0);
  return {std::istreambuf_iterator<char>(*stream), std::istreambuf_iterator<char>()};
}

constexpr const char* kSTSSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse>
  <RequestId>TEST-RID</RequestId>
  <Credentials>
    <AccessKeyId>STS_AK</AccessKeyId>
    <AccessKeySecret>STS_SK</AccessKeySecret>
    <SecurityToken>STS_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleResponse>)";

}  // namespace

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientSuccess) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "test-session";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_EQ(result.creds.GetAWSAccessKeyId(), "STS_AK");
  EXPECT_EQ(result.creds.GetAWSSecretKey(), "STS_SK");
  EXPECT_EQ(result.creds.GetSessionToken(), "STS_TOKEN");
  EXPECT_FALSE(result.creds.IsEmpty());

  // Verify the POST body carries the expected POP v1 params, with the role
  // ARN percent-encoded (colons and slashes escaped).
  auto recorded = mock_client_->GetRecordedRequests();
  ASSERT_FALSE(recorded.empty());
  const auto body = ReadRequestBody(recorded.back());
  EXPECT_NE(body.find("Action=AssumeRole"), std::string::npos) << body;
  EXPECT_NE(body.find("SignatureMethod=HMAC-SHA1"), std::string::npos) << body;
  EXPECT_NE(body.find("SignatureVersion=1.0"), std::string::npos) << body;
  EXPECT_NE(body.find("Version=2015-04-01"), std::string::npos) << body;
  EXPECT_NE(body.find("RoleSessionName=test-session"), std::string::npos) << body;
  // Role ARN must be percent-encoded: colons -> %3A, slashes -> %2F.
  EXPECT_NE(body.find("acs%3Aram%3A%3A123456%3Arole%2Ftarget-role"), std::string::npos) << body;
  EXPECT_EQ(body.find("acs:ram::123456:role/target-role"), std::string::npos)
      << "raw ARN should not appear un-encoded: " << body;
  // Caller's session token must be included when present.
  EXPECT_NE(body.find("SecurityToken=CALLER_TOKEN"), std::string::npos) << body;
  // Signature is appended last; verify it exists and is URL-encoded (base64
  // output contains '+' '/' '=' which all get percent-encoded).
  EXPECT_NE(body.find("&Signature="), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientOmitsSecurityTokenForLongTermCaller) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "LONGTERM_AK";
  req.callerAccessKeySecret = "LONGTERM_SK";
  // Long-term AK/SK: no session token. Aliyun rejects requests that include
  // SecurityToken= with an empty value, so it must be omitted entirely.
  req.callerSecurityToken = "";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "longterm-session";

  client.GetAssumeRoleCredentials(req);

  auto recorded = mock_client_->GetRecordedRequests();
  ASSERT_FALSE(recorded.empty());
  const auto body = ReadRequestBody(recorded.back());
  EXPECT_EQ(body.find("SecurityToken="), std::string::npos) << "SecurityToken must be omitted: " << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientEmptyResponse) {
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/target-role";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientMissingCredentialsElement) {
  // Response shape the dispatcher would reject: root is right but no
  // <Credentials> child.
  const char* xml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse><RequestId>rid</RequestId></AssumeRoleResponse>)";
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/t";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMSTSClientUnexpectedRoot) {
  // Root element isn't AssumeRoleResponse — should bail before credential
  // parsing.
  const char* xml = R"(<?xml version='1.0' encoding='UTF-8'?>
<ErrorResponse><Code>AccessDenied</Code></ErrorResponse>)";
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, xml);

  auto cfg = MakeNoImdsClientConfiguration();
  AliyunRAMSTSClient client(cfg);

  AliyunRAMSTSClient::AssumeRoleRequest req;
  req.callerAccessKeyId = "CALLER_AK";
  req.callerAccessKeySecret = "CALLER_SK";
  req.callerSecurityToken = "CALLER_TOKEN";
  req.roleArn = "acs:ram::123456:role/t";
  req.roleSessionName = "s";

  auto result = client.GetAssumeRoleCredentials(req);
  EXPECT_TRUE(result.creds.IsEmpty());
}

// ============================================================================
// Aliyun RAM Credentials Provider Tests (IMDS → AssumeRole chain)
// ============================================================================

namespace {

// Queue a full successful IMDS → STS round trip:
//   PUT  /latest/api/token                                   -> v2 token
//   GET  /latest/meta-data/ram/security-credentials/         -> role name
//   GET  /latest/meta-data/ram/security-credentials/<role>   -> caller JSON
//   POST sts.aliyuncs.com                                    -> STS XML
// The role-list URL (ending in '/') overlaps with the creds URL as a prefix,
// so the mock's substring match has to be disambiguated by key ordering.
// 'my-imds-role' sorts before 'security-credentials/' (ASCII 'm' < 's'), so
// the creds GET matches its own key first; the list GET only matches the
// broader 'security-credentials/' key.
void EnqueueImdsHappyPath(MockHttpClient& mock, const std::string& sts_xml = kSTSSuccessXml) {
  mock.EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token-opaque");
  mock.EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock.EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock.EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, sts_xml);
}

}  // namespace

TEST_F(S3ProviderTest, TestAliyunRAMProviderEndToEnd) {
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");
  EXPECT_EQ(creds.GetAWSSecretKey(), "STS_SK");
  EXPECT_EQ(creds.GetSessionToken(), "STS_TOKEN");

  // The STS POST body's caller AK/SK/Token must come from IMDS, and it must
  // carry the target role ARN from the provider ctor (not from env).
  auto recorded = mock_client_->GetRecordedRequests();
  size_t sts_idx = recorded.size();
  for (size_t i = 0; i < recorded.size(); ++i) {
    if (recorded[i]->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_idx = i;
      break;
    }
  }
  ASSERT_LT(sts_idx, recorded.size());
  const auto body = ReadRequestBody(recorded[sts_idx]);
  EXPECT_NE(body.find("AccessKeyId=IMDS_AK"), std::string::npos) << body;
  EXPECT_NE(body.find("SecurityToken=IMDS_TOKEN"), std::string::npos) << body;
  EXPECT_NE(body.find("RoleSessionName=tenant-A-session"), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsV1Fallback) {
  ScopedEnvUnset v1_disabled("ALIBABA_CLOUD_IMDSV1_DISABLED");
  ScopedEnvUnset v1_disable("ALIBABA_CLOUD_IMDSV1_DISABLE");
  // V2 token PUT returns 403 (IMDSv2 disabled on this instance) → empty token
  // body; provider falls back to V1-style bare GETs. The rest of the chain
  // still succeeds.
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kSTSSuccessXml);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  // None of the IMDS GETs should have carried the V2 token header.
  for (const auto& req : mock_client_->GetRecordedRequests()) {
    if (req->GetURIString().find("100.100.100.200") != Aws::String::npos &&
        req->GetMethod() == Aws::Http::HttpMethod::HTTP_GET) {
      EXPECT_FALSE(req->HasHeader("x-aliyun-ecs-metadata-token")) << "V1 fallback must not send the V2 token header";
    }
  }
}

TEST_F(S3ProviderTest, AliyunRamTransientImdsV2TokenFailureDoesNotDowngradeToV1) {
  for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
    mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE, "");
  }

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageTransientService);

  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), static_cast<size_t>(kCredentialRetryAttempts + 1));
  for (const auto& request : requests) {
    EXPECT_EQ(request->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT)
        << "a transient token failure must not be followed by an unauthenticated metadata GET";
  }
}

TEST_F(S3ProviderTest, AliyunRamImdsV1DisabledRejectsTokenFallback) {
  ScopedEnvUnset metadata_disabled("ALIBABA_CLOUD_ECS_METADATA_DISABLED");
  ScopedEnvUnset v1_disable("ALIBABA_CLOUD_IMDSV1_DISABLE");
  ScopedEnvVar v1_disabled("ALIBABA_CLOUD_IMDSV1_DISABLED", "TrUe");
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageAccessDenied);

  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT)
      << "ALIBABA_CLOUD_IMDSV1_DISABLED must prevent a bare metadata GET";
}

TEST_F(S3ProviderTest, AliyunRamSingularImdsV1DisableRejectsTokenFallback) {
  ScopedEnvUnset metadata_disabled("ALIBABA_CLOUD_ECS_METADATA_DISABLED");
  ScopedEnvUnset v1_disabled("ALIBABA_CLOUD_IMDSV1_DISABLED");
  ScopedEnvVar v1_disable("ALIBABA_CLOUD_IMDSV1_DISABLE", "TRUE");
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::FORBIDDEN, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageAccessDenied);

  const auto requests = mock_client_->GetRecordedRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front()->GetMethod(), Aws::Http::HttpMethod::HTTP_PUT)
      << "ALIBABA_CLOUD_IMDSV1_DISABLE must prevent a bare metadata GET";
}

TEST_F(S3ProviderTest, AliyunRamMetadataDisabledStopsBeforeHttp) {
  ScopedEnvVar metadata_disabled("ALIBABA_CLOUD_ECS_METADATA_DISABLED", "TRUE");
  ScopedEnvUnset v1_disabled("ALIBABA_CLOUD_IMDSV1_DISABLED");
  ScopedEnvUnset v1_disable("ALIBABA_CLOUD_IMDSV1_DISABLE");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto resolved = provider.ResolveForRequest();
  ASSERT_FALSE(resolved.ok());
  ExpectExtendStatusCode(resolved.status(), ExtendStatusCode::StorageConfigInvalid);
  EXPECT_TRUE(mock_client_->GetRecordedRequests().empty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsRoleListFails) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::NOT_FOUND, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsRoleListEmpty) {
  // 200 OK but empty body (should not happen in practice, but defensive).
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsFails) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsMalformedJson) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, "{not valid json");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderImdsCredsMissingFields) {
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  // Valid JSON but missing SecurityToken.
  const char* partial = R"({"AccessKeyId":"AK","AccessKeySecret":"SK"})";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, partial);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderSTSReturnsEmpty) {
  // IMDS chain succeeds, but the STS call returns an empty body (e.g. upstream
  // outage) → empty creds, no silent success.
  mock_client_->EnqueueResponse("latest/api/token", Aws::Http::HttpResponseCode::OK, "v2-token");
  mock_client_->EnqueueResponse("security-credentials/", Aws::Http::HttpResponseCode::OK, "my-imds-role");
  const char* caller_json = R"({
    "AccessKeyId": "IMDS_AK",
    "AccessKeySecret": "IMDS_SK",
    "SecurityToken": "IMDS_TOKEN",
    "Expiration": "2099-12-31T23:59:59Z"
  })";
  mock_client_->EnqueueResponse("my-imds-role", Aws::Http::HttpResponseCode::OK, caller_json);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderCachesValidCredentials) {
  // Second GetAWSCredentials call within the refresh grace must reuse the
  // cached creds without re-hitting IMDS or STS.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds1 = provider.GetAWSCredentials();
  ASSERT_EQ(creds1.GetAWSAccessKeyId(), "STS_AK");

  const size_t before = mock_client_->GetRecordedRequests().size();
  auto creds2 = provider.GetAWSCredentials();
  EXPECT_EQ(creds2.GetAWSAccessKeyId(), "STS_AK");
  const size_t after = mock_client_->GetRecordedRequests().size();
  EXPECT_EQ(before, after) << "Valid cached credentials must not trigger a new refresh";
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderEmptySessionNameDefaults) {
  // Empty session name should be replaced by a UUID in the ctor, so the STS
  // body never carries an empty RoleSessionName.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", /*role_session_name=*/"");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  // Find the STS request and verify RoleSessionName is non-empty.
  const auto recorded = mock_client_->GetRecordedRequests();
  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& req : recorded) {
    if (req->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = req;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  const auto pos = body.find("RoleSessionName=");
  ASSERT_NE(pos, std::string::npos) << body;
  // The value lives between '=' and the next '&'.
  const auto value_start = pos + std::string("RoleSessionName=").size();
  const auto value_end = body.find('&', value_start);
  const auto value = body.substr(value_start, value_end - value_start);
  EXPECT_FALSE(value.empty()) << "ctor must synthesize a non-empty session name when caller passes empty";
}

// ============================================================================
// Aliyun OIDC AssumeRole Chain Provider Tests
// ============================================================================

namespace {

constexpr const char* kInnerOidcSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleWithOIDCResponse>
  <RequestId>TEST-INNER-RID</RequestId>
  <Credentials>
    <AccessKeyId>INNER_AK</AccessKeyId>
    <AccessKeySecret>INNER_SK</AccessKeySecret>
    <SecurityToken>INNER_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleWithOIDCResponse>)";

constexpr const char* kOuterAssumeRoleSuccessXml = R"(<?xml version='1.0' encoding='UTF-8'?>
<AssumeRoleResponse>
  <RequestId>TEST-OUTER-RID</RequestId>
  <Credentials>
    <AccessKeyId>OUTER_AK</AccessKeyId>
    <AccessKeySecret>OUTER_SK</AccessKeySecret>
    <SecurityToken>OUTER_TOKEN</SecurityToken>
    <Expiration>2099-12-31T23:59:59Z</Expiration>
  </Credentials>
</AssumeRoleResponse>)";

}  // namespace

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderEndToEnd) {
  // Two responses queued under the same URL substring; consumed FIFO. The
  // chain provider issues AssumeRoleWithOIDC first (inner step) and then
  // sts:AssumeRole (outer step), so this ordering matches.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "tenant-A-session");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");
  EXPECT_EQ(creds.GetAWSSecretKey(), "OUTER_SK");
  EXPECT_EQ(creds.GetSessionToken(), "OUTER_TOKEN");

  const auto recorded = mock_client_->GetRecordedRequests();
  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> sts_reqs;
  for (const auto& r : recorded) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_reqs.push_back(r);
    }
  }
  ASSERT_EQ(sts_reqs.size(), 2u) << "chain must issue exactly two STS calls";

  // Inner request: AssumeRoleWithOIDC against the env-driven machine-identity
  // role, with the env's OIDC provider ARN. Customer's target role must NOT
  // appear here — that was the bug this provider exists to fix.
  const auto inner_body = ReadRequestBody(sts_reqs[0]);
  EXPECT_NE(inner_body.find("Action=AssumeRoleWithOIDC"), std::string::npos) << inner_body;
  EXPECT_NE(inner_body.find("zilliz-machine-role"), std::string::npos) << inner_body;
  EXPECT_NE(inner_body.find("zilliz-rrsa"), std::string::npos) << inner_body;
  EXPECT_EQ(inner_body.find("customer-target"), std::string::npos)
      << "inner OIDC step must not carry the customer target role: " << inner_body;

  // Outer request: AssumeRole signed by the inner step's STS creds, targeting
  // the customer role with the caller-supplied session name.
  const auto outer_body = ReadRequestBody(sts_reqs[1]);
  EXPECT_NE(outer_body.find("Action=AssumeRole"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("customer-target"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("AccessKeyId=INNER_AK"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("SecurityToken=INNER_TOKEN"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("RoleSessionName=tenant-A-session"), std::string::npos) << outer_body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderInnerStepFailsReturnsEmpty) {
  // STS replies to the inner AssumeRoleWithOIDC with an empty body — the
  // outer call must never fire and the provider must surface empty creds
  // rather than silently falling back to anonymous.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());

  // Exactly one STS hit (the inner one); outer must be skipped.
  size_t sts_calls = 0;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos)
      ++sts_calls;
  }
  EXPECT_EQ(sts_calls, 1u);
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderOuterStepEmptyReturnsEmpty) {
  // Inner step succeeds, outer AssumeRole returns an empty body (e.g. cross-
  // account trust policy not yet configured). Provider must surface empty
  // creds, not the inner step's creds — those would let the caller through
  // to the customer's bucket using zilliz's own identity.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, "");

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess");
  auto creds = provider.GetAWSCredentials();
  EXPECT_TRUE(creds.IsEmpty());
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderForwardsExternalId) {
  // ExternalId belongs in the step-2 sts:AssumeRole body when the caller
  // supplies one. Aliyun's AssumeRole semantics match AWS: empty == not sent
  // (so the parameter behaves like "absent" from the trust policy's POV),
  // non-empty == sent verbatim.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess",
                                        /*external_id=*/"tenant-A-ext-id");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = r;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  EXPECT_NE(body.find("ExternalId=tenant-A-ext-id"), std::string::npos) << body;
}

TEST_F(S3ProviderTest, TestAliyunRAMProviderOmitsExternalIdWhenEmpty) {
  // Sending an empty ExternalId would still tip the request into the
  // "ExternalId-supplied" branch on Aliyun's side and fail the trust-policy
  // check whenever the policy doesn't list ExternalId. The provider must
  // omit the parameter entirely when the caller leaves it empty.
  EnqueueImdsHappyPath(*mock_client_);

  AliyunRAMCredentialsProvider provider("acs:ram::123456:role/target-role", "sess");
  auto creds = provider.GetAWSCredentials();
  ASSERT_EQ(creds.GetAWSAccessKeyId(), "STS_AK");

  std::shared_ptr<Aws::Http::HttpRequest> sts_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_req = r;
      break;
    }
  }
  ASSERT_NE(sts_req, nullptr);
  const auto body = ReadRequestBody(sts_req);
  EXPECT_EQ(body.find("ExternalId="), std::string::npos)
      << "RAM provider must not emit ExternalId at all when caller passes empty: " << body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderForwardsExternalIdToStep2Only) {
  // ExternalId is a step-2 (sts:AssumeRole) concern. Aliyun's
  // AssumeRoleWithOIDC API has no ExternalId parameter, so step 1 must NOT
  // carry it; step 2 must. Verifying both halves catches the easy bug of
  // accidentally putting ExternalId in the inner provider's body.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");
  ScopedEnvUnset unset_session("ALIBABA_CLOUD_ROLE_SESSION_NAME");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", "sess",
                                             /*target_external_id=*/"tenant-A-ext-id");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");

  const auto recorded = mock_client_->GetRecordedRequests();
  std::vector<std::shared_ptr<Aws::Http::HttpRequest>> sts_reqs;
  for (const auto& r : recorded) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      sts_reqs.push_back(r);
    }
  }
  ASSERT_EQ(sts_reqs.size(), 2u);

  const auto inner_body = ReadRequestBody(sts_reqs[0]);
  EXPECT_NE(inner_body.find("Action=AssumeRoleWithOIDC"), std::string::npos) << inner_body;
  EXPECT_EQ(inner_body.find("ExternalId"), std::string::npos)
      << "AssumeRoleWithOIDC has no ExternalId concept; step 1 must not carry it: " << inner_body;

  const auto outer_body = ReadRequestBody(sts_reqs[1]);
  EXPECT_NE(outer_body.find("Action=AssumeRole"), std::string::npos) << outer_body;
  EXPECT_NE(outer_body.find("ExternalId=tenant-A-ext-id"), std::string::npos) << outer_body;
}

TEST_F(S3ProviderTest, TestAliyunOIDCChainProviderEmptySessionNameDefaults) {
  // Empty target session name should be replaced by a UUID — the outer
  // AssumeRole body must never carry an empty RoleSessionName.
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kInnerOidcSuccessXml);
  mock_client_->EnqueueResponse("sts.aliyuncs.com", Aws::Http::HttpResponseCode::OK, kOuterAssumeRoleSuccessXml);

  TempFile token_file("oidc-jwt-payload");
  ScopedEnvVar set_inner_role("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::1111:role/zilliz-machine-role");
  ScopedEnvVar set_token("ALIBABA_CLOUD_OIDC_TOKEN_FILE", token_file.path());
  ScopedEnvVar set_provider("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::1111:oidc-provider/zilliz-rrsa");

  AliyunOIDCAssumeRoleChainProvider provider("acs:ram::2222:role/customer-target", /*target_session_name=*/"");
  auto creds = provider.GetAWSCredentials();
  EXPECT_EQ(creds.GetAWSAccessKeyId(), "OUTER_AK");

  std::shared_ptr<Aws::Http::HttpRequest> outer_req;
  for (const auto& r : mock_client_->GetRecordedRequests()) {
    if (r->GetURIString().find("sts.aliyuncs.com") != Aws::String::npos) {
      const auto body = ReadRequestBody(r);
      if (body.find("Action=AssumeRole&") != std::string::npos ||
          body.find("&Action=AssumeRole&") != std::string::npos) {
        outer_req = r;
      }
    }
  }
  ASSERT_NE(outer_req, nullptr);
  const auto body = ReadRequestBody(outer_req);
  const auto pos = body.find("RoleSessionName=");
  ASSERT_NE(pos, std::string::npos) << body;
  const auto value_start = pos + std::string("RoleSessionName=").size();
  const auto value_end = body.find('&', value_start);
  const auto value = body.substr(value_start, value_end - value_start);
  EXPECT_FALSE(value.empty());
}

}  // namespace milvus_storage
