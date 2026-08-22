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

#pragma once

#include <chrono>
#include <exception>
#include <memory>
#include <string_view>

#include <arrow/status.h>
#include <arrow/result.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>

namespace milvus_storage {

/// How many times a credential client retries before giving up.
///
/// One number for every provider, because the layering is: this library makes a
/// bounded attempt and then reports what happened; deciding whether to try
/// again later belongs to the caller, which knows what the operation was worth.
/// Before this was uniform the same repository ran three different policies --
/// IMDS retried zero times, the web identity clients three, and the remaining
/// STS clients whatever AWS_MAX_ATTEMPTS happened to say -- so how long a
/// credential failure took to surface depended on which cloud you were on.
inline constexpr long kCredentialRetryAttempts = 3;
inline constexpr long kCredentialConnectTimeoutMs = 2000;
inline constexpr long kCredentialRequestTimeoutMs = 5000;
// IMDS is link-local and answers in milliseconds on a healthy instance, so it
// gets a much tighter cap: with the retry budget above, 5s per attempt made a
// single failing refresh hold the credentials lock for tens of seconds, which
// is exactly what the original short caps existed to prevent.
inline constexpr long kImdsRequestTimeoutMs = 1500;

/// A request-local credential resolution gate.
///
/// `AWSCredentialsProvider::GetAWSCredentials()` is the SDK's interface: it
/// hands back credentials and nothing else. A provider that could not reach
/// STS therefore answers with an empty set and the reason dies with the call.
///
/// ResolveForRequest returns the credentials and their typed failure as one
/// value. S3's Standard and CRT holders call it before handing a client to an
/// object request, so an expired credential whose refresh failed never falls
/// through as an anonymous request and later masquerades as AccessDenied.
class RequestCredentialsResolver {
  public:
  virtual ~RequestCredentialsResolver() = default;

  [[nodiscard]] virtual arrow::Result<Aws::Auth::AWSCredentials> ResolveForRequest() = 0;
};

/// Validate a short-lived credential set as one indivisible value.
///
/// AWSCredentials::IsEmpty() only checks that both AK and SK are empty. It
/// accepts half-filled credentials and treats a missing expiration as "never
/// expires", which is unsafe for STS responses. Every temporary credential
/// cache must pass this check before publishing the new value.
arrow::Status ValidateTemporaryCredentials(const Aws::Auth::AWSCredentials& credentials, std::string_view source);

/// How long a caller keeps trying to resolve credentials before deciding the
/// attempt belongs to someone else.
inline constexpr std::chrono::milliseconds kCredentialResolutionBudget{100};

/// Whether a caller that entered credential resolution at `started` still has
/// budget left to make an attempt of its own.
///
/// Asked AFTER the writer lock is acquired, because acquiring it can take as
/// long as another thread's entire reload.
///
/// The double-check that precedes this already covers the case where that
/// thread SUCCEEDED -- the credentials are fresh, everyone returns them, and
/// nobody reloads. What it cannot cover is that thread having FAILED: the
/// credentials are still empty, so the condition is false for every queued
/// caller and they each replay the same outage in series, with the queue as
/// long as the thread pool. Asking "do I still have budget" ends that without
/// anyone having to record whose failure it was.
bool CredentialAttemptStillWorthMaking(std::chrono::steady_clock::time_point started);

/// Send a request through a raw HttpClient with the shared retry budget.
///
/// `HttpClient::MakeRequest` does NOT consume
/// `ClientConfiguration::retryStrategy` -- the SDK's retry loop lives in
/// `AWSHttpResourceClient::GetResourceWithAWSWebServiceResult`, and the call
/// sites that talk to IMDS or to Huawei's second stage bypass it. Setting the
/// strategy on the config was therefore decoration on those paths: they stayed
/// one-shot no matter what the number said.
///
/// Retries only what a retry can fix -- no response at all, or a 5xx/429/408
/// from the service. A 4xx means the request was refused on its merits and
/// repeating it just delays the answer.
std::shared_ptr<Aws::Http::HttpResponse> MakeRequestWithCredentialRetry(
    Aws::Http::HttpClient& client, const std::shared_ptr<Aws::Http::HttpRequest>& request);

/// Classify a credential request from the HTTP outcome it actually got.
///
/// The point of separating these is NOT to make the SDK retry again -- the
/// credential clients already have their own bounded retry, and by the time a
/// status reaches a caller that budget is spent. It is to let the layer above
/// re-run the whole operation later. Collapsing every cause into
/// StorageAccessDenied made that impossible: the code is SYSTEM-category, it
/// lands on segcore's ConfigInvalid, and the operation is never attempted
/// again -- so a metadata service that was restarting failed a query exactly as
/// permanently as a role ARN that does not exist.
///
/// Order matters: the SDK's sentinels sit inside ordinary numeric ranges.
/// NO_RESPONSE is 444 and the network timeouts are 598/599, so classifying by
/// range first would call a dead network a bad configuration.
arrow::Status ClassifyCredentialHttpFailure(Aws::Http::HttpResponseCode code, std::string_view what);

/// The deployment is missing something this process needs: a token file that is
/// not mounted, no role attached to the instance. Not retryable, and naming it
/// points whoever is paged at the thing to change.
arrow::Status MakeCredentialConfigError(std::string_view what);

/// The service answered, and we could not use the answer: an unparseable body,
/// a response with no credentials in it.
///
/// Deliberately left unclassified rather than mapped to AccessDenied. It is
/// neither a transport fault we can wait out nor a refusal on the merits, and
/// dressing it as either would be a guess -- the conservative unclassified
/// landing is non-retryable without claiming to know why.
arrow::Status MakeCredentialResponseError(std::string_view what);

/// Turn a dependency exception into the Status contract used by credential
/// resolvers. Allocation failures remain Arrow OOM statuses, so the direct
/// C++ boundary can report MemAllocateFailed. Other exceptions are deliberately
/// unclassified: no HTTP or provider verdict was observed.
arrow::Status MakeCredentialOutOfMemoryError(std::string_view what);
arrow::Status MakeCredentialExceptionError(std::string_view what, const std::exception& exception);
arrow::Status MakeCredentialUnknownExceptionError(std::string_view what);

}  // namespace milvus_storage
