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

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <aws/core/Aws.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/S3Errors.h>

#include <arrow/filesystem/filesystem.h>
#include <arrow/status.h>
#include "milvus-storage/common/log.h"
#include <arrow/util/print.h>
#include <arrow/util/string.h>

#include "milvus-storage/common/extend_status.h"

namespace milvus_storage {
namespace fs {
namespace internal {

// XXX Should we expose this at some point?
enum class S3Backend { Amazon, Minio, Other };

enum class S3ResourceKind : uint8_t {
  /// The operation does not identify one resource kind, or the caller has not
  /// carried that fact. Generic RESOURCE_NOT_FOUND keeps the conservative
  /// object-missing behavior for compatibility.
  Unknown,
  /// The request addressed an object/key.
  Object,
  /// The request addressed the bucket itself (HeadBucket, ListObjects, etc.).
  Bucket,
  /// The request addressed multipart-upload state rather than an object that
  /// could independently be absent. A generic 404 here is reported as
  /// NoSuchUpload; a vanished bucket is indistinguishable without extra IO,
  /// which is deliberately not spent.
  MultipartUpload,
};

/// \brief Which kind of resource the failed request addressed.
struct S3ErrorProvenance {
  constexpr explicit S3ErrorProvenance(S3ResourceKind resource_kind = S3ResourceKind::Unknown)
      : resource_kind(resource_kind) {}

  /// \brief Which resource a bodyless/generic 404 was answering about.
  ///
  /// S3's RESOURCE_NOT_FOUND value is used for both a missing object and a
  /// missing bucket. Use only the failed operation's resource kind; do not
  /// issue a follow-up request to disambiguate an already failed operation.
  S3ResourceKind resource_kind;
};

/// \brief Keep a message useful without letting it grow unbounded.
///
/// The tail of an S3 error message is whatever the endpoint felt like sending
/// -- a proxy in front of a misconfigured store answers a signed GET with a
/// full HTML page, and that lands verbatim in a Status that is then logged,
/// wrapped and carried across the FFI boundary. The head is where the useful
/// part is.
inline std::string BoundedDetail(const std::string& text, size_t limit = 512) {
  if (text.size() <= limit) {
    return text;
  }
  return text.substr(0, limit) + "...(+" + ::arrow::internal::ToChars(text.size() - limit) + " bytes)";
}

inline bool IsExpiredCredentialError(std::string_view exception_name) {
  return exception_name == "ExpiredToken" || exception_name == "ExpiredTokenException" ||
         exception_name == "TokenRefreshRequired";
}

/// \brief Does the service error name identify rejected credentials or access?
///
/// DeleteObjects reports per-key failures as strings inside an otherwise
/// successful response, so those errors bypass AWSError and its enum/HTTP
/// classification. Keep the accepted spellings in one place so request-level
/// and per-key failures retain the same structured verdict.
inline bool IsAccessDeniedErrorName(std::string_view exception_name) {
  return IsExpiredCredentialError(exception_name) || exception_name == "AccessDenied" ||
         exception_name == "InvalidToken" || exception_name == "InvalidAccessKeyId" ||
         exception_name == "SignatureDoesNotMatch" || exception_name == "InvalidSecurity";
}

// Detect the S3 backend type from the S3 server's response headers
inline S3Backend DetectS3Backend(const Aws::Http::HeaderValueCollection& headers) {
  const auto it = headers.find("server");
  if (it != headers.end()) {
    const auto& value = std::string_view(it->second);
    if (value.find("AmazonS3") != std::string::npos) {
      return S3Backend::Amazon;
    }
    if (value.find("MinIO") != std::string::npos) {
      return S3Backend::Minio;
    }
  }
  return S3Backend::Other;
}

template <typename Error>
inline S3Backend DetectS3Backend(const Aws::Client::AWSError<Error>& error) {
  return DetectS3Backend(error.GetResponseHeaders());
}

template <typename ErrorType>
inline std::optional<std::string> BucketRegionFromError(const Aws::Client::AWSError<ErrorType>& error) {
  if constexpr (std::is_same_v<ErrorType, Aws::S3::S3Errors>) {
    const auto& headers = error.GetResponseHeaders();
    const auto it = headers.find("x-amz-bucket-region");
    if (it != headers.end()) {
      const std::string region(it->second.begin(), it->second.end());
      return region;
    }
  }
  return std::nullopt;
}

/// \brief Does this error mean the object/key is absent?
///
/// RESOURCE_NOT_FOUND is intentionally accepted here because object HEADs on
/// AWS are bodyless and commonly lose the more precise NoSuchKey spelling. It
/// is ambiguous with a missing bucket; the ambiguity is accepted and reported
/// as a missing key -- resolving it would cost an extra RPC on every miss.
inline bool IsObjectNotFound(const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  const auto error_type = error.GetErrorType();
  return error_type == Aws::S3::S3Errors::NO_SUCH_KEY || error_type == Aws::S3::S3Errors::RESOURCE_NOT_FOUND;
}

/// \brief Does a bucket-level operation say the bucket is absent?
inline bool IsBucketNotFound(const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  const auto error_type = error.GetErrorType();
  return error_type == Aws::S3::S3Errors::NO_SUCH_BUCKET || error_type == Aws::S3::S3Errors::RESOURCE_NOT_FOUND;
}

/// \brief Is the response unambiguously about a missing bucket even on an
/// object-level operation?
inline bool IsExplicitBucketNotFound(const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  return error.GetErrorType() == Aws::S3::S3Errors::NO_SUCH_BUCKET;
}

inline bool IsAlreadyExists(const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  const auto error_type = error.GetErrorType();
  return (error_type == Aws::S3::S3Errors::BUCKET_ALREADY_EXISTS ||
          error_type == Aws::S3::S3Errors::BUCKET_ALREADY_OWNED_BY_YOU);
}

inline std::string S3ErrorToString(Aws::S3::S3Errors error_type) {
  switch (error_type) {
#define S3_ERROR_CASE(NAME)     \
  case Aws::S3::S3Errors::NAME: \
    return #NAME;

    S3_ERROR_CASE(INCOMPLETE_SIGNATURE)
    S3_ERROR_CASE(INTERNAL_FAILURE)
    S3_ERROR_CASE(INVALID_ACTION)
    S3_ERROR_CASE(INVALID_CLIENT_TOKEN_ID)
    S3_ERROR_CASE(INVALID_PARAMETER_COMBINATION)
    S3_ERROR_CASE(INVALID_QUERY_PARAMETER)
    S3_ERROR_CASE(INVALID_PARAMETER_VALUE)
    S3_ERROR_CASE(MISSING_ACTION)
    S3_ERROR_CASE(MISSING_AUTHENTICATION_TOKEN)
    S3_ERROR_CASE(MISSING_PARAMETER)
    S3_ERROR_CASE(OPT_IN_REQUIRED)
    S3_ERROR_CASE(REQUEST_EXPIRED)
    S3_ERROR_CASE(SERVICE_UNAVAILABLE)
    S3_ERROR_CASE(THROTTLING)
    S3_ERROR_CASE(VALIDATION)
    S3_ERROR_CASE(ACCESS_DENIED)
    S3_ERROR_CASE(RESOURCE_NOT_FOUND)
    S3_ERROR_CASE(UNRECOGNIZED_CLIENT)
    S3_ERROR_CASE(MALFORMED_QUERY_STRING)
    S3_ERROR_CASE(SLOW_DOWN)
    S3_ERROR_CASE(REQUEST_TIME_TOO_SKEWED)
    S3_ERROR_CASE(INVALID_SIGNATURE)
    S3_ERROR_CASE(SIGNATURE_DOES_NOT_MATCH)
    S3_ERROR_CASE(INVALID_ACCESS_KEY_ID)
    S3_ERROR_CASE(REQUEST_TIMEOUT)
    S3_ERROR_CASE(NETWORK_CONNECTION)
    S3_ERROR_CASE(UNKNOWN)
    S3_ERROR_CASE(BUCKET_ALREADY_EXISTS)
    S3_ERROR_CASE(BUCKET_ALREADY_OWNED_BY_YOU)
    // The following is the most recent addition to S3Errors
    // and is not supported yet for some versions of the SDK
    // that Apache Arrow is using. This is not a big deal
    // since this error will happen only in very specialized
    // settings and we will print the correct numerical error
    // code as per the "default" case down below. We should
    // put it back once the SDK has been upgraded in all
    // Apache Arrow build configurations.
    // S3_ERROR_CASE(INVALID_OBJECT_STATE)
    S3_ERROR_CASE(NO_SUCH_BUCKET)
    S3_ERROR_CASE(NO_SUCH_KEY)
    S3_ERROR_CASE(NO_SUCH_UPLOAD)
    S3_ERROR_CASE(OBJECT_ALREADY_IN_ACTIVE_TIER)
    S3_ERROR_CASE(OBJECT_NOT_IN_ACTIVE_TIER)

#undef S3_ERROR_CASE
    default:
      return "[code " + ::arrow::internal::ToChars(static_cast<int>(error_type)) + "]";
  }
}

inline std::optional<arrow::Status> tryMakeClassifiedExtendArrowError(Aws::S3::S3Errors error_type,
                                                                      Aws::Http::HttpResponseCode response_code,
                                                                      const std::string& message,
                                                                      std::string_view exception_name,
                                                                      const std::string& extra_info,
                                                                      S3ErrorProvenance provenance) {
  // GCP OAuth token resolution happens inside the process-global HTTP client
  // factory, after the S3 call has begun. These internal XML names carry its
  // typed Result across the AWS SDK marshalling boundary without pretending a
  // malformed token response was an access-control decision.
  if (exception_name == "LoonGcpCredentialConfigInvalid") {
    return MakeExtendError(ExtendStatusCode::StorageConfigInvalid, message, extra_info);
  }
  if (exception_name == "LoonGcpCredentialResponseInvalid") {
    return arrow::Status::IOError(message);
  }

  // Some SDK/parser combinations preserve the service's exception name while
  // normalizing the enum to ACCESS_DENIED; others leave it UNKNOWN. By the
  // time this error escapes the SDK, its credential provider has already been
  // asked for credentials. This layer cannot promise that replaying the same
  // operation will obtain a different token (explicit and default-chain
  // credentials may be static), so report the observed authentication failure
  // instead of inventing a retry guarantee.
  if (IsAccessDeniedErrorName(exception_name)) {
    return MakeExtendError(ExtendStatusCode::StorageAccessDenied, message, extra_info);
  }

  switch (error_type) {
    case Aws::S3::S3Errors::NO_SUCH_UPLOAD:
      // A dead upload id cannot be repaired by replaying the same multipart
      // request. The owner of the upload may choose to start a new one.
      return MakeExtendError(ExtendStatusCode::StorageNoSuchUpload, message, extra_info);
    case Aws::S3::S3Errors::NO_SUCH_BUCKET:
      // Split out of the not-found group on purpose. A missing bucket is not
      // data loss and re-reading the manifest cannot conjure one -- the
      // deployment points somewhere that does not exist, which is a
      // configuration fix, not an object to go looking for.
      return MakeExtendError(ExtendStatusCode::StorageBucketNotFound, message, extra_info);
    case Aws::S3::S3Errors::NO_SUCH_KEY:
      return MakeExtendError(ExtendStatusCode::StorageNotFound, message, extra_info);
    case Aws::S3::S3Errors::RESOURCE_NOT_FOUND:
      if (provenance.resource_kind == S3ResourceKind::Bucket) {
        return MakeExtendError(ExtendStatusCode::StorageBucketNotFound, message, extra_info);
      }
      if (provenance.resource_kind == S3ResourceKind::MultipartUpload) {
        return MakeExtendError(ExtendStatusCode::StorageNoSuchUpload, message, extra_info);
      }
      return MakeExtendError(ExtendStatusCode::StorageNotFound, message, extra_info);
    case Aws::S3::S3Errors::ACCESS_DENIED:
    case Aws::S3::S3Errors::INVALID_ACCESS_KEY_ID:
    case Aws::S3::S3Errors::SIGNATURE_DOES_NOT_MATCH:
      return MakeExtendError(ExtendStatusCode::StorageAccessDenied, message, extra_info);
    case Aws::S3::S3Errors::UNKNOWN:
      // The SDK models only a fraction of what a store can answer, and
      // everything it does not model arrives here nameless. Credential
      // failures are the ones that matter: ExpiredToken and its relatives left
      // as bare IOErrors reached an external-table caller as LOON_ARROW_ERROR
      // -- a generic internal failure -- when the caller's own credentials
      // were the thing at fault and they are the only one who can fix them.
      //
      // Matched on the error NAME as well as the status, because
      // S3-compatible stores spell 401/403 inconsistently while the token
      // names are stable across them.
      if (IsAccessDeniedErrorName(exception_name) || response_code == Aws::Http::HttpResponseCode::UNAUTHORIZED ||
          response_code == Aws::Http::HttpResponseCode::FORBIDDEN) {
        return MakeExtendError(ExtendStatusCode::StorageAccessDenied, message, extra_info);
      }
      switch (response_code) {
        case Aws::Http::HttpResponseCode::PRECONDITION_FAILED:
          return MakeExtendError(ExtendStatusCode::StoragePreConditionFailed, message, extra_info);
        case Aws::Http::HttpResponseCode::CONFLICT:
          // 409 is overloaded. A lost race (conditional write, concurrent
          // commit) is Conflict: re-read state and re-submit. BucketNotEmpty
          // and InvalidBucketState are 409s too, but they are answers about
          // the bucket, not about a race -- replaying cannot change either.
          // Exclude the known non-conflict shapes rather than allowlist the
          // races, so the transaction commit path keeps its Conflict verdict
          // across S3-compatible stores whose race spellings differ.
          if (exception_name == "BucketNotEmpty" || exception_name == "InvalidBucketState") {
            return std::nullopt;
          }
          return MakeExtendError(ExtendStatusCode::StorageConflict, message, extra_info);
        default:
          return std::nullopt;
      }
    default:
      return std::nullopt;
  }
}

template <typename ErrorType>
std::optional<arrow::Status> tryMakeRetryableExtendArrowError(const Aws::Client::AWSError<ErrorType>& error,
                                                              Aws::S3::S3Errors error_type,
                                                              const std::string& message,
                                                              const std::string& extra_info = {}) {
  switch (error_type) {
    case Aws::S3::S3Errors::INTERNAL_FAILURE:
      return MakeExtendError(ExtendStatusCode::StorageTransientService, message, extra_info);
    case Aws::S3::S3Errors::REQUEST_TIMEOUT:
      return MakeExtendError(ExtendStatusCode::StorageTransientTimeout, message, extra_info);
    case Aws::S3::S3Errors::THROTTLING:
    case Aws::S3::S3Errors::SLOW_DOWN:
      return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, message, extra_info);
    case Aws::S3::S3Errors::SERVICE_UNAVAILABLE:
      return MakeExtendError(ExtendStatusCode::StorageTransientService, message, extra_info);
    case Aws::S3::S3Errors::NETWORK_CONNECTION:
      return MakeExtendError(ExtendStatusCode::StorageTransientNetwork, message, extra_info);
    default:
      break;
  }

  switch (error.GetResponseCode()) {
    case Aws::Http::HttpResponseCode::REQUEST_TIMEOUT:
      return MakeExtendError(ExtendStatusCode::StorageTransientTimeout, message, extra_info);
    case Aws::Http::HttpResponseCode::TOO_MANY_REQUESTS:
      return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, message, extra_info);
    case Aws::Http::HttpResponseCode::INTERNAL_SERVER_ERROR:
    case Aws::Http::HttpResponseCode::BAD_GATEWAY:
    case Aws::Http::HttpResponseCode::SERVICE_UNAVAILABLE:
    case Aws::Http::HttpResponseCode::GATEWAY_TIMEOUT:
      return MakeExtendError(ExtendStatusCode::StorageTransientService, message, extra_info);
    default:
      break;
  }

  const auto exception_name = error.GetExceptionName();
  if (exception_name == "SlowDown" || exception_name == "SlowDownWrite") {
    return MakeExtendError(ExtendStatusCode::StorageTransientThrottling, message, extra_info);
  }
  if (exception_name == "XMinioServerNotInitialized") {
    return MakeExtendError(ExtendStatusCode::StorageTransientService, message, extra_info);
  }

  // ShouldRetry is the SDK retry policy, not a diagnosis. If no error type,
  // HTTP status or backend exception name identifies the condition, leave the
  // status unclassified instead of inventing a network failure.
  return std::nullopt;
}

template <typename ErrorType>
arrow::Status ErrorToStatus(const std::string& prefix,
                            const std::string& operation,
                            const Aws::Client::AWSError<ErrorType>& error,
                            S3ErrorProvenance provenance,
                            const std::optional<std::string>& region = std::nullopt) {
  const std::string exception_name(error.GetExceptionName().data(), error.GetExceptionName().size());
  const std::string aws_message(error.GetMessage().data(), error.GetMessage().size());
  const std::string bounded_operation = BoundedDetail(operation, 128);
  const std::string bounded_exception = exception_name.empty() ? "(none)" : BoundedDetail(exception_name, 128);

  // XXX Handle fine-grained error types
  // See
  // https://sdk.amazonaws.com/cpp/api/LATEST/namespace_aws_1_1_s3.html#ae3f82f8132b619b6e91c88a9f1bde371
  // CoreErrors and S3Errors share a numeric space only by accident: the S3
  // enum continues where the core one stops, so casting a core code into it
  // names whichever S3 error happens to sit at that number. Answer core errors
  // on their own terms before that cast.
  //
  // The bound is SERVICE_EXTENSION_START_RANGE (128), which is where the SDK
  // stops numbering core errors and service enums begin -- not VALIDATION (14),
  // which is merely one core code among many. Gating on VALIDATION made this
  // whole block unreachable for the three codes it exists for
  // (UNRECOGNIZED_CLIENT=17, INVALID_SIGNATURE=21, MEMORY_ALLOCATION=26): they
  // sit above it, so every one still fell through to the S3 cast.
  if (static_cast<int>(error.GetErrorType()) <
      static_cast<int>(Aws::Client::CoreErrors::SERVICE_EXTENSION_START_RANGE)) {
    const auto core = static_cast<Aws::Client::CoreErrors>(error.GetErrorType());
    // Carries the prefix for the same reason the classified messages below do:
    // this is where the bucket and key are named.
    std::string core_message = BoundedDetail(prefix, 768) + "AWS core error " + bounded_exception + " during " +
                               bounded_operation + " operation: " + BoundedDetail(aws_message);
    core_message = BoundedDetail(core_message, 2048);
    const std::string core_extra_info =
        BoundedDetail("operation=" + bounded_operation +
                          " core_error=" + ::arrow::internal::ToChars(static_cast<int>(error.GetErrorType())) +
                          " exception=" + bounded_exception +
                          " http_status=" + ::arrow::internal::ToChars(static_cast<int>(error.GetResponseCode())),
                      512);
    // The provider was already consulted before the signed request failed.
    // Without a provider-specific invalidation signal, an immediate replay may
    // reuse the same expired credential, so keep this as authentication failure.
    if (IsExpiredCredentialError(exception_name)) {
      return MakeExtendError(ExtendStatusCode::StorageAccessDenied, core_message, core_extra_info);
    }
    switch (core) {
      case Aws::Client::CoreErrors::MEMORY_ALLOCATION:
        return arrow::Status::OutOfMemory(core_message);
      case Aws::Client::CoreErrors::UNRECOGNIZED_CLIENT:
      case Aws::Client::CoreErrors::MISSING_AUTHENTICATION_TOKEN:
      case Aws::Client::CoreErrors::INVALID_CLIENT_TOKEN_ID:
      case Aws::Client::CoreErrors::INVALID_SIGNATURE:
        // Credentials the deployment supplied are not accepted. An operator
        // fixes this; it is not the caller's request and not our bug.
        return MakeExtendError(ExtendStatusCode::StorageAccessDenied, core_message, core_extra_info);
      default:
        break;
    }
  }

  auto error_type = static_cast<Aws::S3::S3Errors>(error.GetErrorType());
  std::stringstream ss;
  ss << S3ErrorToString(error_type);
  if (error_type == Aws::S3::S3Errors::UNKNOWN) {
    ss << " (HTTP status " << static_cast<int>(error.GetResponseCode()) << ")";
  }

  // Possibly an error due to wrong region configuration from client and bucket.
  std::optional<std::string> wrong_region_msg = std::nullopt;
  if (region.has_value()) {
    const auto maybe_region = BucketRegionFromError(error);
    if (maybe_region.has_value() && maybe_region.value() != region.value()) {
      wrong_region_msg = " Looks like the configured region is '" + BoundedDetail(region.value(), 128) +
                         "' while the bucket is located in '" + BoundedDetail(maybe_region.value(), 128) + "'.";
    }
  }
  // Built ONCE, with the prefix, and used for every arm below.
  //
  // The prefix is where the bucket and the key are ("When reading information
  // for key 'k' in bucket 'b': "). It used to be glued on only by the trailing
  // IOError, so every CLASSIFIED status -- the ones that reach an operator with
  // a verdict attached, and the ones that get paged on -- named the operation
  // and nothing it operated on: "AccessDenied during HeadObject", with no way
  // to tell which bucket rejected which key.
  std::string message = BoundedDetail(prefix, 768) + "AWS Error " + ss.str() + " during " + bounded_operation +
                        " operation: " + BoundedDetail(aws_message) + wrong_region_msg.value_or("");
  message = BoundedDetail(message, 2048);
  LOG_STORAGE_WARNING_ << message;

  // extra_info used to be the message a second time, which cost a copy of an
  // unbounded string per failure and told a reader nothing the message had not
  // already said. What is here instead is the part a human message renders
  // badly: the store's own name for the condition (S3-compatible backends put
  // their real answer there and nowhere else -- "SlowDownWrite",
  // "XMinioServerNotInitialized", "ExpiredToken") and the HTTP status the
  // verdict was derived from.
  std::string extra_info = "operation=" + bounded_operation + " s3_error=" + ss.str() +
                           " exception=" + bounded_exception +
                           " http_status=" + ::arrow::internal::ToChars(static_cast<int>(error.GetResponseCode()));
  if (wrong_region_msg.has_value()) {
    const auto actual_region = BucketRegionFromError(error).value();
    extra_info += " configured_region=" + BoundedDetail(region.value(), 128) +
                  " actual_region=" + BoundedDetail(actual_region, 128);
  }
  extra_info = BoundedDetail(extra_info, 512);

  if (auto classified_status = tryMakeClassifiedExtendArrowError(
          error_type, error.GetResponseCode(), message, std::string_view(exception_name), extra_info, provenance);
      classified_status.has_value()) {
    return classified_status.value();
  }

  if (auto retryable_status = tryMakeRetryableExtendArrowError(error, error_type, message, extra_info);
      retryable_status.has_value()) {
    return retryable_status.value();
  }

  // x-amz-bucket-region is the service's authoritative answer. When it differs
  // from the region used to sign the request, replaying the same configuration
  // cannot help.
  //
  // Checked LAST, not first. The header is an attribute of the response, not a
  // diagnosis of the failure: a store is free to echo it on a 503 or a 429, and
  // a request that was throttled while pointed at the wrong region was still
  // throttled. Judging it ahead of the identified condition let one true fact
  // erase a sharper one, which is the downgrade R2.3 forbids. The mismatch
  // stays in the message and extra_info on every arm above, so a misconfigured
  // region is still visible in whichever verdict actually applies. It becomes
  // THE verdict only when nothing else identified the failure -- which is the
  // redirect/malformed-authorization shape it exists for, since the SDK reports
  // those as UNKNOWN with a 301/400 that no other arm claims.
  if (wrong_region_msg.has_value()) {
    return MakeExtendError(ExtendStatusCode::StorageConfigInvalid, message, extra_info);
  }

  // ShouldRetry is an SDK policy decision, not an observed cause.  If the
  // error type, HTTP response, transport condition, or backend name above did
  // not identify the condition, leave it unclassified instead of publishing
  // either a transient or permanent verdict derived from that policy.
  return arrow::Status::IOError(message);
}

template <typename ErrorType, typename... Args>
arrow::Status ErrorToStatus(const std::tuple<Args&...>& prefix,
                            const std::string& operation,
                            const Aws::Client::AWSError<ErrorType>& error,
                            S3ErrorProvenance provenance) {
  std::stringstream ss;
  ::arrow::internal::PrintTuple(&ss, prefix);
  return ErrorToStatus(ss.str(), operation, error, provenance);
}

template <typename ErrorType>
arrow::Status ErrorToStatus(const std::string& operation,
                            const Aws::Client::AWSError<ErrorType>& error,
                            S3ErrorProvenance provenance) {
  return ErrorToStatus(std::string(), operation, error, provenance);
}

template <typename AwsResult, typename Error>
arrow::Status OutcomeToStatus(const std::string& prefix,
                              const std::string& operation,
                              const Aws::Utils::Outcome<AwsResult, Error>& outcome,
                              S3ErrorProvenance provenance) {
  if (outcome.IsSuccess()) {
    return arrow::Status::OK();
  } else {
    return ErrorToStatus(prefix, operation, outcome.GetError(), provenance);
  }
}

template <typename AwsResult, typename Error, typename... Args>
arrow::Status OutcomeToStatus(const std::tuple<Args&...>& prefix,
                              const std::string& operation,
                              const Aws::Utils::Outcome<AwsResult, Error>& outcome,
                              S3ErrorProvenance provenance) {
  if (outcome.IsSuccess()) {
    return arrow::Status::OK();
  } else {
    return ErrorToStatus(prefix, operation, outcome.GetError(), provenance);
  }
}

template <typename AwsResult, typename Error>
arrow::Status OutcomeToStatus(const std::string& operation,
                              const Aws::Utils::Outcome<AwsResult, Error>& outcome,
                              S3ErrorProvenance provenance) {
  return OutcomeToStatus(std::string(), operation, outcome, provenance);
}

template <typename AwsResult, typename Error>
arrow::Result<AwsResult> OutcomeToResult(const std::string& operation,
                                         Aws::Utils::Outcome<AwsResult, Error> outcome,
                                         S3ErrorProvenance provenance) {
  if (outcome.IsSuccess()) {
    return std::move(outcome).GetResultWithOwnership();
  } else {
    return ErrorToStatus(operation, outcome.GetError(), provenance);
  }
}

inline Aws::String ToAwsString(const std::string& s) {
  // Direct construction of Aws::String from std::string doesn't work because
  // it uses a specific Allocator class.
  return Aws::String(s.begin(), s.end());
}

inline std::string_view FromAwsString(const Aws::String& s) { return {s.data(), s.length()}; }

inline Aws::String ToURLEncodedAwsString(const std::string& s) { return Aws::Utils::StringUtils::URLEncode(s.data()); }

inline arrow::fs::TimePoint FromAwsDatetime(const Aws::Utils::DateTime& dt) {
  return std::chrono::time_point_cast<std::chrono::nanoseconds>(dt.UnderlyingTimestamp());
}

// A connect retry strategy with a controlled max duration.

class ConnectRetryStrategy : public Aws::Client::RetryStrategy {
  public:
  static const int32_t kDefaultRetryInterval = 200;     /* milliseconds */
  static const int32_t kDefaultMaxRetryDuration = 6000; /* milliseconds */

  explicit ConnectRetryStrategy(int32_t retry_interval = kDefaultRetryInterval,
                                int32_t max_retry_duration = kDefaultMaxRetryDuration)
      : retry_interval_(retry_interval), max_retry_duration_(max_retry_duration) {}

  bool ShouldRetry(const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
                   long attempted_retries) const override {  // NOLINT runtime/int
    // This class is itself an AWS SDK retry policy. It may consult the SDK's
    // eligibility flag, but that flag must never escape through ErrorToStatus
    // as a storage diagnosis or public retry contract.
    const auto exception_name = error.GetExceptionName();
    const bool retry_eligible = error.ShouldRetry() || exception_name == "SlowDownWrite" ||
                                exception_name == "SlowDown" || exception_name == "XMinioServerNotInitialized";
    if (!retry_eligible) {
      return false;
    }
    return attempted_retries * retry_interval_ < max_retry_duration_;
  }

  long CalculateDelayBeforeNextRetry(  // NOLINT runtime/int
      const Aws::Client::AWSError<Aws::Client::CoreErrors>& error,
      long attempted_retries) const override {  // NOLINT runtime/int
    return retry_interval_;
  }

  protected:
  int32_t retry_interval_;
  int32_t max_retry_duration_;
};

}  // namespace internal
}  // namespace fs
}  // namespace milvus_storage
