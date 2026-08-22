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

#include "milvus-storage/filesystem/s3/provider/credential_resolution.h"

#include <string>
#include <thread>

#include <aws/core/http/HttpResponse.h>
#include <arrow/util/string_builder.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/common/log.h"

namespace milvus_storage {

namespace {

// Only what a retry can fix. A 4xx is the service refusing the request on its
// merits; repeating it changes nothing and delays the real answer.
bool WorthRetrying(const std::shared_ptr<Aws::Http::HttpResponse>& response) {
  if (response == nullptr) {
    return true;
  }
  const auto code = static_cast<int>(response->GetResponseCode());
  if (code <= 0 || code == 408 || code == 429 || code == 444) {
    return true;
  }
  return code >= 500;
}

}  // namespace

arrow::Status ValidateTemporaryCredentials(const Aws::Auth::AWSCredentials& credentials, std::string_view source) {
  const std::string prefix(source);
  if (credentials.GetAWSAccessKeyId().empty()) {
    return MakeCredentialResponseError(prefix + " returned temporary credentials without an access key");
  }
  if (credentials.GetAWSSecretKey().empty()) {
    return MakeCredentialResponseError(prefix + " returned temporary credentials without a secret key");
  }
  if (credentials.GetSessionToken().empty()) {
    return MakeCredentialResponseError(prefix + " returned temporary credentials without a session token");
  }

  const auto expiration = credentials.GetExpiration();
  const auto never_expires = (std::chrono::system_clock::time_point::max)();
  if (!expiration.WasParseSuccessful() || expiration.UnderlyingTimestamp() == never_expires) {
    return MakeCredentialResponseError(prefix + " returned temporary credentials without a valid expiration");
  }
  if (expiration <= Aws::Utils::DateTime::Now()) {
    return MakeCredentialResponseError(prefix + " returned expired temporary credentials");
  }
  return arrow::Status::OK();
}

bool CredentialAttemptStillWorthMaking(std::chrono::steady_clock::time_point started) {
  return std::chrono::steady_clock::now() - started < kCredentialResolutionBudget;
}

// Put a request body back at the start so the next attempt sends it again.
//
// Resending the same HttpRequest resends a stream that the previous attempt
// already read to the end, so a retried POST would go out with an empty body
// and be rejected on its contents rather than retried on its merits. Curl's
// own CURLOPT_SEEKFUNCTION does not help: that is for curl replaying a request
// it is already in the middle of, and it is never called when MakeRequest is
// entered again.
//
// Returns false when the body cannot be rewound, which is the signal to stop
// retrying rather than send something malformed. Every credential POST in this
// repository carries an in-memory StringStream, so this succeeds in practice.
bool RewindBodyForRetry(const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  const auto& body = request->GetContentBody();
  if (body == nullptr) {
    return true;
  }
  body->clear();
  body->seekg(0, std::ios_base::beg);
  return !body->fail();
}

std::shared_ptr<Aws::Http::HttpResponse> MakeRequestWithCredentialRetry(
    Aws::Http::HttpClient& client, const std::shared_ptr<Aws::Http::HttpRequest>& request) {
  std::shared_ptr<Aws::Http::HttpResponse> response;
  for (long attempt = 0; attempt <= kCredentialRetryAttempts; ++attempt) {
    response = client.MakeRequest(request);
    if (!WorthRetrying(response)) {
      return response;
    }
    if (attempt == kCredentialRetryAttempts) {
      break;
    }
    if (!RewindBodyForRetry(request)) {
      LOG_STORAGE_WARNING_ << "Not retrying a credential request whose body cannot be rewound";
      break;
    }
    // Exponential backoff from 25ms. The added sleep stays well under a second;
    // each I/O attempt remains bounded separately by the client's connect and
    // request timeouts. This runs while a caller holds the credentials lock.
    std::this_thread::sleep_for(std::chrono::milliseconds(25L << attempt));
  }
  return response;
}

arrow::Status ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode code, std::string_view what) {
  using Aws::Http::HttpResponseCode;
  switch (code) {
    case HttpResponseCode::REQUEST_NOT_MADE:
    case HttpResponseCode::NO_RESPONSE:
      // Nothing reached the service, so nothing about the deployment has been
      // proven wrong. A retry is the only way to find out.
      return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, what);
    case HttpResponseCode::REQUEST_TIMEOUT:
    case HttpResponseCode::NETWORK_READ_TIMEOUT:
    case HttpResponseCode::NETWORK_CONNECT_TIMEOUT:
      return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientTimeout, what);
    case HttpResponseCode::TOO_MANY_REQUESTS:
    case HttpResponseCode::BANDWIDTH_LIMIT_EXCEEDED:
      return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientThrottling, what);
    case HttpResponseCode::UNAUTHORIZED:
    case HttpResponseCode::FORBIDDEN:
      // The service identified us and said no: a trust policy that does not
      // name us, an identity token it will not accept. This is the one
      // credential failure that really is an access decision.
      return MakeExtendErrorMsg(ExtendStatusCode::StorageAccessDenied, what);
    default:
      break;
  }

  const auto value = static_cast<int>(code);
  if (value <= 0) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientNetwork, what);
  }
  if (value >= 500) {
    return MakeExtendErrorMsg(ExtendStatusCode::StorageTransientService, what);
  }
  if (value >= 400) {
    // Refused on the merits, but not as an access decision -- a malformed
    // request, an unknown role. Something in the deployment produced it.
    return MakeExtendErrorMsg(ExtendStatusCode::StorageConfigInvalid, what);
  }
  // A 2xx/3xx that still yielded no credentials.
  return MakeCredentialResponseError(what);
}

arrow::Status MakeCredentialConfigError(std::string_view what) {
  return MakeExtendErrorMsg(ExtendStatusCode::StorageConfigInvalid, what);
}

arrow::Status MakeCredentialResponseError(std::string_view what) {
  // No ExtendStatusDetail on purpose: unclassified lands in the conservative
  // non-retryable bucket without asserting a cause we have not established.
  return arrow::Status::IOError(what);
}

arrow::Status MakeCredentialOutOfMemoryError(std::string_view what) {
  return arrow::Status::OutOfMemory(what);
}

arrow::Status MakeCredentialExceptionError(std::string_view what, const std::exception& exception) {
  return MakeCredentialResponseError(arrow::util::StringBuilder(what, ": ", exception.what()));
}

arrow::Status MakeCredentialUnknownExceptionError(std::string_view what) {
  return MakeCredentialResponseError(arrow::util::StringBuilder(what, ": unknown exception"));
}

}  // namespace milvus_storage
