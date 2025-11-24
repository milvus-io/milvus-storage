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

#include "milvus-storage/filesystem/s3/s3_client.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <string>
#include <vector>

#include <arrow/util/logging.h>
#include <arrow/result.h>
#include <arrow/util/thread_pool.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/io/interfaces.h>

#include <aws/core/Aws.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/xml/XmlSerializer.h>
#include <aws/core/internal/AWSHttpResourceClient.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/ObjectCannedACL.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/UploadPartRequest.h>

#include "milvus-storage/common/path_util.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_internal.h"
#include "milvus-storage/filesystem/s3/util_internal.h"

using ::arrow::Result;
using ::arrow::Status;

using ::Aws::Client::AWSError;
using ::Aws::S3::S3Errors;
using ::milvus_storage::S3Options;
using ::milvus_storage::fs::internal::ConnectRetryStrategy;
using ::milvus_storage::fs::internal::FromAwsString;
using ::milvus_storage::fs::internal::ToAwsString;

namespace milvus_storage {

namespace S3Model = Aws::S3::Model;
static inline constexpr auto kBucketRegionHeaderName = "x-amz-bucket-region";

inline arrow::Status ErrorS3Finalized() { return arrow::Status::Invalid("S3 subsystem is finalized"); }

template <typename ObjectRequest>
struct ObjectMetadataSetter {
  using Setter = std::function<Status(const std::string& value, ObjectRequest* req)>;

  static std::unordered_map<std::string, Setter> GetSetters() {
    return {{"ACL", CannedACLSetter()},
            {"Cache-Control", StringSetter(&ObjectRequest::SetCacheControl)},
            {"Content-Type", ContentTypeSetter()},
            {"Content-Language", StringSetter(&ObjectRequest::SetContentLanguage)},
            {"Expires", DateTimeSetter(&ObjectRequest::SetExpires)}};
  }

  private:
  static Setter StringSetter(void (ObjectRequest::*req_method)(Aws::String&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(ToAwsString(v));
      return arrow::Status::OK();
    };
  }

  static Setter DateTimeSetter(void (ObjectRequest::*req_method)(Aws::Utils::DateTime&&)) {
    return [req_method](const std::string& v, ObjectRequest* req) {
      (req->*req_method)(Aws::Utils::DateTime(v.data(), Aws::Utils::DateFormat::ISO_8601));
      return arrow::Status::OK();
    };
  }

  static Setter CannedACLSetter() {
    return [](const std::string& v, ObjectRequest* req) {
      ARROW_ASSIGN_OR_RAISE(auto acl, ParseACL(v));
      req->SetACL(acl);
      return arrow::Status::OK();
    };
  }

  /** We need a special setter here and can not use `StringSetter` because for e.g. the
   * `PutObjectRequest`, the setter is located in the base class (instead of the concrete
   * class). */
  static Setter ContentTypeSetter() {
    return [](const std::string& str, ObjectRequest* req) {
      req->SetContentType(str);
      return arrow::Status::OK();
    };
  }

  static arrow::Result<S3Model::ObjectCannedACL> ParseACL(const std::string& v) {
    if (v.empty()) {
      return S3Model::ObjectCannedACL::NOT_SET;
    }
    auto acl = S3Model::ObjectCannedACLMapper::GetObjectCannedACLForName(ToAwsString(v));
    if (acl == S3Model::ObjectCannedACL::NOT_SET) {
      // XXX This actually never happens, as the AWS SDK dynamically
      // expands the enum range using Aws::GetEnumOverflowContainer()
      return arrow::Status::Invalid("Invalid S3 canned ACL: '", v, "'");
    }
    return acl;
  }
};

class WrappedRetryStrategy : public Aws::Client::RetryStrategy {
  public:
  explicit WrappedRetryStrategy(const std::shared_ptr<S3RetryStrategy>& s3_retry_strategy)
      : s3_retry_strategy_(s3_retry_strategy) {}

  bool ShouldRetry(const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
                   long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return s3_retry_strategy_->ShouldRetry(detail, static_cast<int64_t>(attempted_retries));
  }

  long CalculateDelayBeforeNextRetry(  // NOLINT runtime/int
      const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
      long attempted_retries) const override {  // NOLINT runtime/int
    S3RetryStrategy::AWSErrorDetail detail = ErrorToDetail(error);
    return static_cast<long>(  // NOLINT runtime/int
        s3_retry_strategy_->CalculateDelayBeforeNextRetry(detail, static_cast<int64_t>(attempted_retries)));
  }

  private:
  template <typename ErrorType>
  static S3RetryStrategy::AWSErrorDetail ErrorToDetail(const Aws::Client::AWSError<ErrorType>& error) {
    S3RetryStrategy::AWSErrorDetail detail;
    detail.error_type = static_cast<int>(error.GetErrorType());
    detail.message = std::string(FromAwsString(error.GetMessage()));
    detail.exception_name = std::string(FromAwsString(error.GetExceptionName()));
    detail.should_retry = error.ShouldRetry();
    return detail;
  }

  std::shared_ptr<S3RetryStrategy> s3_retry_strategy_;
};

// ------------ Implementation of S3Client ------------
std::string S3Client::GetBucketRegionFromHeaders(const Aws::Http::HeaderValueCollection& headers) {
  const auto it = headers.find(ToAwsString(kBucketRegionHeaderName));
  if (it != headers.end()) {
    return std::string(FromAwsString(it->second));
  }
  return std::string();
}

template <typename ErrorType>
arrow::Result<std::string> S3Client::GetBucketRegionFromError(const std::string& bucket,
                                                              const Aws::Client::AWSError<ErrorType>& error) {
  std::string region = GetBucketRegionFromHeaders(error.GetResponseHeaders());
  if (!region.empty()) {
    return region;
  } else if (error.GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND) {
    return arrow::Status::IOError("Bucket '", bucket, "' not found");
  } else {
    return arrow::Status::IOError("When resolving region for bucket: ", bucket);
  }
}

arrow::Result<std::string> S3Client::GetBucketRegion(const std::string& bucket,
                                                     const S3Model::HeadBucketRequest& request) {
  auto uri = GeneratePresignedUrl(request.GetBucket(),
                                  /*key=*/"", Aws::Http::HttpMethod::HTTP_HEAD);
  // NOTE: The signer region argument isn't passed here, as there's no easy
  // way of computing it (the relevant method is private).
  auto outcome = MakeRequest(uri, request, Aws::Http::HttpMethod::HTTP_HEAD, Aws::Auth::SIGV4_SIGNER);
  if (!outcome.IsSuccess()) {
    return GetBucketRegionFromError(bucket, outcome.GetError());
  }
  std::string region = GetBucketRegionFromHeaders(outcome.GetResult().GetHeaderValueCollection());
  if (!region.empty()) {
    return region;
  } else if (outcome.GetResult().GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND) {
    return arrow::Status::IOError("Bucket '", request.GetBucket(), "' not found");
  } else {
    return arrow::Status::IOError("When resolving region for bucket '", request.GetBucket(),
                                  "': missing 'x-amz-bucket-region' header in response");
  }
}

arrow::Result<std::string> S3Client::GetBucketRegion(const std::string& bucket) {
  S3Model::HeadBucketRequest req;
  req.SetBucket(ToAwsString(bucket));
  return GetBucketRegion(bucket, req);
}

S3Model::CompleteMultipartUploadOutcome S3Client::CompleteMultipartUploadWithErrorFixup(
    S3Model::CompleteMultipartUploadRequest&& request) const {
  // CompletedMultipartUpload can return a 200 OK response with an error
  // encoded in the response body, in which case we should either retry
  // or propagate the error to the user (see
  // https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html).
  //
  // Unfortunately the AWS SDK doesn't detect such situations but lets them
  // return successfully (see https://github.com/aws/aws-sdk-cpp/issues/658).
  //
  // We work around the issue by registering a DataReceivedEventHandler
  // which parses the XML response for embedded errors.

  std::optional<AWSError<Aws::Client::CoreErrors>> aws_error;
  metrics_->IncrementMultiPartUploadFinished();

  auto handler = [&](const Aws::Http::HttpRequest* http_req, Aws::Http::HttpResponse* http_resp,
                     long long) {  // NOLINT runtime/int
    auto& stream = http_resp->GetResponseBody();
    const auto pos = stream.tellg();
    const auto doc = Aws::Utils::Xml::XmlDocument::CreateFromXmlStream(stream);
    // Rewind stream for later
    stream.clear();
    stream.seekg(pos);

    if (doc.WasParseSuccessful()) {
      auto root = doc.GetRootElement();
      if (!root.IsNull()) {
        // Detect something that looks like an abnormal CompletedMultipartUpload
        // response.
        if (root.GetName() != "CompleteMultipartUploadResult" || !root.FirstChild("Error").IsNull() ||
            !root.FirstChild("Errors").IsNull()) {
          // Make sure the error marshaller doesn't see a 200 OK
          http_resp->SetResponseCode(Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR);
          aws_error = GetErrorMarshaller()->Marshall(*http_resp);
          // Rewind stream for later
          stream.clear();
          stream.seekg(pos);
        }
      }
    }
  };

  request.SetDataReceivedEventHandler(std::move(handler));

  // We don't have access to the configured AWS retry strategy
  // (m_retryStrategy is a private member of AwsClient), so don't use that.
  std::unique_ptr<Aws::Client::RetryStrategy> retry_strategy;
  if (s3_retry_strategy_) {
    retry_strategy.reset(new WrappedRetryStrategy(s3_retry_strategy_));
  } else {
    // Note that DefaultRetryStrategy, unlike StandardRetryStrategy,
    // has empty definitions for RequestBookkeeping() and GetSendToken(),
    // which simplifies the code below.
    retry_strategy.reset(new Aws::Client::DefaultRetryStrategy());
  }

  for (int32_t retries = 0;; retries++) {
    aws_error.reset();
    auto outcome = Aws::S3::S3Client::S3Client::CompleteMultipartUpload(request);
    if (!outcome.IsSuccess()) {
      // Error returned in HTTP headers (or client failure)
      return outcome;
    }
    if (!aws_error.has_value()) {
      // Genuinely successful outcome
      return outcome;
    }

    const bool should_retry = retry_strategy->ShouldRetry(*aws_error, retries);

    ARROW_LOG(WARNING) << "CompletedMultipartUpload got error embedded in a 200 OK response: "
                       << aws_error->GetExceptionName() << " (\"" << aws_error->GetMessage()
                       << "\"), retry = " << should_retry;

    if (!should_retry) {
      break;
    }
    const auto delay = std::chrono::milliseconds(retry_strategy->CalculateDelayBeforeNextRetry(*aws_error, retries));
    std::this_thread::sleep_for(delay);
  }

  DCHECK(aws_error.has_value());
  auto s3_error = AWSError<S3Errors>(std::move(aws_error).value());
  return S3Model::CompleteMultipartUploadOutcome(std::move(s3_error));
}

// Metrics related functions
S3Model::CreateMultipartUploadOutcome S3Client::CreateMultipartUpload(
    const Aws::S3::Model::CreateMultipartUploadRequest& request) const {
  metrics_->IncrementMultiPartUploadCreated();
  auto outcome = Aws::S3::S3Client::CreateMultipartUpload(request);
  if (!outcome.IsSuccess()) {
    metrics_->IncrementFailedCount();
  }
  return outcome;
}

S3Model::UploadPartOutcome S3Client::UploadPart(const Aws::S3::Model::UploadPartRequest& request) const {
  metrics_->IncrementUploadCount();
  auto outcome = Aws::S3::S3Client::UploadPart(request);
  if (!outcome.IsSuccess()) {
    metrics_->IncrementFailedCount();
  } else {
    metrics_->IncrementUploadBytes(request.GetContentLength());
  }
  return outcome;
}

S3Model::PutObjectOutcome S3Client::PutObject(const Aws::S3::Model::PutObjectRequest& request) const {
  metrics_->IncrementUploadCount();
  auto outcome = Aws::S3::S3Client::PutObject(request);
  if (!outcome.IsSuccess()) {
    metrics_->IncrementFailedCount();
  } else {
    metrics_->IncrementUploadBytes(request.GetContentLength());
  }
  return outcome;
}

S3Model::GetObjectOutcome S3Client::GetObject(const Aws::S3::Model::GetObjectRequest& request) const {
  metrics_->IncrementDownloadCount();
  auto outcome = Aws::S3::S3Client::GetObject(request);
  if (!outcome.IsSuccess()) {
    metrics_->IncrementFailedCount();
  } else {
    metrics_->IncrementDownloadBytes(outcome.GetResult().GetContentLength());
  }
  return outcome;
}

std::shared_ptr<S3ClientMetrics> S3Client::GetMetrics() const { return metrics_; }
// ------------ Implementation of S3Client End ------------

// ------------ Implementation of S3ClientHolder ------------
S3Client* S3ClientLock::get() { return client_.get(); }
S3Client* S3ClientLock::operator->() { return client_.get(); }

// Move this S3ClientLock into a temporary instance
//
// It is counter-intuitive, but lock ordering issues can happen even
// with a shared mutex locked in shared mode.
// The reason is that locking again in shared mode can block while
// there are threads waiting to take the lock in exclusive mode.
// Therefore, we should avoid obtaining the S3ClientLock when
// we already have it locked.
//
// This methods helps by moving the S3ClientLock into a temporary
// that is immediately destroyed so the lock will be released as
// soon as we are done making the call to the underlying client.
//
// (see GH-36523)
S3ClientLock S3ClientLock::Move() { return std::move(*this); }
// ------------ Implementation of S3ClientHolder End ------------

// ------------ Implementation of S3ClientHolder ------------
S3ClientHolder::S3ClientHolder(std::weak_ptr<S3ClientFinalizer> finalizer, std::shared_ptr<S3Client> client)
    : finalizer_(std::move(finalizer)), client_(std::move(client)) {}

arrow::Result<S3ClientLock> S3ClientHolder::Lock() {
  std::shared_ptr<S3ClientFinalizer> finalizer;
  std::shared_ptr<S3Client> client;
  {
    std::unique_lock lock(mutex_);
    finalizer = finalizer_.lock();
    client = client_;
  }
  // Do not hold mutex while taking finalizer lock below.
  //
  // Acquiring a shared_mutex in shared mode may block even if not already
  // acquired in exclusive mode, because of pending writers:
  // https://github.com/google/sanitizers/issues/1668#issuecomment-1624985664
  // """It is implementation-defined whether the calling thread acquires
  // the lock when a writer does not hold the lock and there are writers
  // blocked on the lock""".
  //
  // Therefore, we want to avoid potential lock ordering issues
  // even when a shared lock is involved (GH-36523).
  if (!finalizer) {
    return ErrorS3Finalized();
  }

  S3ClientLock client_lock;
  // Lock the finalizer before examining it
  client_lock.lock_ = finalizer->LockShared();
  if (finalizer->finalized_) {
    return ErrorS3Finalized();
  }
  // (the client can be cleared only if finalizer->finalized_ is true)
  DCHECK(client) << "inconsistent S3ClientHolder";
  client_lock.client_ = std::move(client);
  return client_lock;
}

void S3ClientHolder::Finalize() {
  std::shared_ptr<S3Client> client;
  {
    std::unique_lock lock(mutex_);
    client = std::move(client_);
  }
  // Do not hold mutex while ~S3Client potentially runs
}
// ------------ Implementation of S3ClientHolder End ------------

using ClientHolderList = std::vector<std::weak_ptr<S3ClientHolder>>;

arrow::Result<std::shared_ptr<S3ClientHolder>> S3ClientFinalizer::AddClient(std::shared_ptr<S3Client> client) {
  std::unique_lock lock(mutex_);
  if (finalized_) {
    return ErrorS3Finalized();
  }

  auto holder = std::make_shared<S3ClientHolder>(shared_from_this(), std::move(client));

  // Remove expired entries before adding new one
  auto end = std::remove_if(holders_.begin(), holders_.end(),
                            [](std::weak_ptr<S3ClientHolder> holder) { return holder.expired(); });
  holders_.erase(end, holders_.end());
  holders_.emplace_back(holder);
  return holder;
}

void S3ClientFinalizer::Finalize() {
  std::unique_lock lock(mutex_);
  finalized_ = true;

  ClientHolderList finalizing = std::move(holders_);
  lock.unlock();  // avoid lock ordering issue with S3ClientHolder::Finalize

  // Finalize all client holders, such that no S3Client remains alive
  // after this.
  for (auto&& weak_holder : finalizing) {
    auto holder = weak_holder.lock();
    if (holder) {
      holder->Finalize();
    }
  }
}

std::shared_lock<std::shared_mutex> S3ClientFinalizer::LockShared() { return std::shared_lock(mutex_); }

std::shared_ptr<S3ClientFinalizer> GetClientFinalizer() {
  static auto finalizer = std::make_shared<S3ClientFinalizer>();
  return finalizer;
}

arrow::Result<std::shared_ptr<S3ClientHolder>> GetClientHolder(std::shared_ptr<S3Client> client) {
  return GetClientFinalizer()->AddClient(std::move(client));
}

// ------------ Implementation of ClientBuilder ------------
ClientBuilder::ClientBuilder(S3Options options) : options_(std::move(options)) {}

const Aws::Client::ClientConfiguration& ClientBuilder::config() const { return client_config_; }

Aws::Client::ClientConfiguration* ClientBuilder::mutable_config() { return &client_config_; }

const S3Options& ClientBuilder::options() const { return options_; }

arrow::Result<std::shared_ptr<S3ClientHolder>> ClientBuilder::BuildClient(
    std::optional<arrow::io::IOContext> io_context) {
  credentials_provider_ = options_.credentials_provider;
  if (!options_.region.empty()) {
    client_config_.region = ToAwsString(options_.region);
  }
  if (options_.request_timeout > 0) {
    // Use ceil() to avoid setting it to 0 as that probably means no timeout.
    client_config_.requestTimeoutMs = static_cast<long>(ceil(options_.request_timeout * 1000));  // NOLINT runtime/int
  }
  if (options_.connect_timeout > 0) {
    client_config_.connectTimeoutMs = static_cast<long>(ceil(options_.connect_timeout * 1000));  // NOLINT runtime/int
  }

  client_config_.endpointOverride = ToAwsString(options_.endpoint_override);
  if (options_.scheme == "http") {
    client_config_.scheme = Aws::Http::Scheme::HTTP;
    client_config_.verifySSL = false;
  } else if (options_.scheme == "https") {
    client_config_.scheme = Aws::Http::Scheme::HTTPS;
    client_config_.verifySSL = true;
  } else {
    return arrow::Status::Invalid("Invalid S3 connection scheme '", options_.scheme, "'");
  }
  if (options_.retry_strategy) {
    client_config_.retryStrategy = std::make_shared<WrappedRetryStrategy>(options_.retry_strategy);
  } else {
    client_config_.retryStrategy = std::make_shared<ConnectRetryStrategy>();
  }
  if (!arrow::fs::internal::global_options.tls_ca_file_path.empty()) {
    client_config_.caFile = ToAwsString(arrow::fs::internal::global_options.tls_ca_file_path);
    client_config_.verifySSL = false;
  }
  if (!arrow::fs::internal::global_options.tls_ca_dir_path.empty()) {
    client_config_.caPath = ToAwsString(arrow::fs::internal::global_options.tls_ca_dir_path);
  }

  // Set proxy options if provided
  if (!options_.proxy_options.scheme.empty()) {
    if (options_.proxy_options.scheme == "http") {
      client_config_.proxyScheme = Aws::Http::Scheme::HTTP;
    } else if (options_.proxy_options.scheme == "https") {
      client_config_.proxyScheme = Aws::Http::Scheme::HTTPS;
    } else {
      return arrow::Status::Invalid("Invalid proxy connection scheme '", options_.proxy_options.scheme, "'");
    }
  }
  if (!options_.proxy_options.host.empty()) {
    client_config_.proxyHost = ToAwsString(options_.proxy_options.host);
  }
  if (options_.proxy_options.port != -1) {
    client_config_.proxyPort = options_.proxy_options.port;
  }
  if (!options_.proxy_options.username.empty()) {
    client_config_.proxyUserName = ToAwsString(options_.proxy_options.username);
  }
  if (!options_.proxy_options.password.empty()) {
    client_config_.proxyPassword = ToAwsString(options_.proxy_options.password);
  }

  if (io_context) {
    // TODO: Once ARROW-15035 is done we can get rid of the "at least 25" fallback
    client_config_.maxConnections = std::max(io_context->executor()->GetCapacity(), 25);
  }

  client_config_.maxConnections = std::max(client_config_.maxConnections, options_.max_connections);

  const bool use_virtual_addressing = options_.endpoint_override.empty() || options_.force_virtual_addressing;

#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
  client_config_.useVirtualAddressing = use_virtual_addressing;
  auto endpoint_provider = EndpointProviderCache::Instance()->Lookup(client_config_);
  auto client = std::make_shared<S3Client>(credentials_provider_, endpoint_provider, client_config_);
#else
  auto client =
      std::make_shared<S3Client>(credentials_provider_, client_config_,
                                 Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, use_virtual_addressing);
#endif
  client->s3_retry_strategy_ = options_.retry_strategy;
  return GetClientHolder(std::move(client));
}

// ------------ Implementation of ClientBuilder End ------------

}  // namespace milvus_storage