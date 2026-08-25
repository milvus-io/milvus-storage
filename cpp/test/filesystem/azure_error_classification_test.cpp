// Copyright 2025 Zilliz
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

// Pins the Azure -> ExtendStatusCode mapping and, more importantly, where each
// class lands at the segcore boundary. These run without an Azure account: the
// classifier takes the raw HTTP status rather than an SDK exception precisely
// so the taxonomy is testable in CI, where every credential-bearing Azure test
// is skipped.

#include <gtest/gtest.h>

#include <azure/core/exception.hpp>

#include <optional>
#include <string>
#include <string_view>

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/azure/azurefs_internal.h"

namespace milvus_storage::fs::internal {
namespace {

milvus::ErrorCode SegcoreCodeFor(int http_status,
                                 std::string_view error_code = "",
                                 std::string_view reason_phrase = "") {
  auto code = ClassifyAzureError(http_status, error_code, /*transport_failure=*/false, reason_phrase);
  if (!code.has_value()) {
    // Untagged: this is what the caller actually produces -- a plain IOError.
    return ToSegcoreError(arrow::Status::IOError("azure failure")).get_error_code();
  }
  return ToSegcoreErrorCode(*code);
}

}  // namespace

TEST(AzureErrorClassification, TransientStatusesAreRetriable) {
  struct Case {
    int http_status;
    std::string_view error_code;
    ExtendStatusCode expected;
  };
  // The availability-relevant half: before this mapping existed every one of
  // these reached segcore as a permanent StorageError/2044 and was never
  // retried.
  const Case cases[] = {
      {408, "", ExtendStatusCode::StorageTransientTimeout},
      {429, "TooManyRequests", ExtendStatusCode::StorageTransientThrottling},
      {503, "ServerBusy", ExtendStatusCode::StorageTransientThrottling},
      {503, "", ExtendStatusCode::StorageTransientService},
      {500, "", ExtendStatusCode::StorageTransientService},
      {502, "", ExtendStatusCode::StorageTransientService},
      {504, "", ExtendStatusCode::StorageTransientService},
  };

  for (const auto& c : cases) {
    auto code = ClassifyAzureError(c.http_status, c.error_code, /*transport_failure=*/false);
    ASSERT_TRUE(code.has_value()) << c.http_status << " " << c.error_code;
    EXPECT_EQ(*code, c.expected) << c.http_status << " " << c.error_code;
    EXPECT_TRUE(RetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), milvus::StorageTransientError) << c.http_status;
  }

  // A transport failure carries no status at all; it is identified by the
  // exception's dynamic type, not by status == 0.
  auto transport = ClassifyAzureError(0, "", /*transport_failure=*/true);
  ASSERT_TRUE(transport.has_value());
  EXPECT_EQ(*transport, ExtendStatusCode::StorageTransientNetwork);
  EXPECT_TRUE(RetryableForExtendStatusCode(*transport));
}

// The counter-example that makes `transport_failure` load-bearing.
//
// Azure's plain `RequestFailedException(std::string)` also leaves StatusCode at
// None, and PollUntilDone raises exactly that when a copy operation ends in a
// failed/aborted state (see the second catch in Impl::CopyFile). Keying the
// network verdict on `status == 0` would report a copy that definitively failed
// as a retriable network blip -- the precise transient/permanent inversion this
// taxonomy exists to prevent.
TEST(AzureErrorClassification, StatusZeroWithoutTransportIsNotRetriable) {
  auto code = ClassifyAzureError(0, "", /*transport_failure=*/false);
  EXPECT_FALSE(code.has_value());
  EXPECT_EQ(SegcoreCodeFor(0), milvus::StorageError);
  EXPECT_NE(SegcoreCodeFor(0), milvus::StorageTransientError);

  // Same input, but genuinely a transport failure: retriable.
  auto transport = ClassifyAzureError(0, "", /*transport_failure=*/true);
  ASSERT_TRUE(transport.has_value());
  EXPECT_EQ(ToSegcoreErrorCode(*transport), milvus::StorageTransientError);
}

TEST(AzureErrorClassification, NonRetriableStatusesNeverLookTransient) {
  struct Case {
    int http_status;
    std::string_view error_code;
    ExtendStatusCode expected;
    milvus::ErrorCode segcore;
  };
  const Case cases[] = {
      // Not-found is fine-grained: a consumer can tell "data missing" from a
      // generic storage failure, matching what the S3 path already reports.
      {404, "BlobNotFound", ExtendStatusCode::StorageNotFound, milvus::ObjectNotExist},
      // System: the credentials are operator configuration, so this has to
      // reach whoever owns the deployment (segcore 2006) rather than be filed
      // as a generic storage failure (2044).
      {401, "", ExtendStatusCode::StorageAccessDenied, milvus::ConfigInvalid},
      {403, "AuthenticationFailed", ExtendStatusCode::StorageAccessDenied, milvus::ConfigInvalid},
  };

  for (const auto& c : cases) {
    auto code = ClassifyAzureError(c.http_status, c.error_code, /*transport_failure=*/false);
    ASSERT_TRUE(code.has_value()) << c.http_status << " " << c.error_code;
    EXPECT_EQ(*code, c.expected) << c.http_status;
    EXPECT_FALSE(RetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), c.segcore) << c.http_status;
    // A non-retriable failure must never look transient, or a consumer
    // retry-storms a request that can never succeed.
    EXPECT_NE(ToSegcoreErrorCode(*code), milvus::StorageTransientError) << c.http_status;
  }
}

// 412 and 409-already-exists used to sit in the System test above. They are
// Conflict: someone else won a race, and a new re-read/rebase attempt may
// succeed, while replaying the same conditional request fails identically.
//
// That is why Conflict is its own category rather than part of Transient, and
// why these cases need their own test.
TEST(AzureErrorClassification, PreconditionFailuresAreConflictNotSystem) {
  struct Case {
    int http_status;
    std::string_view error_code;
  };
  const Case cases[] = {
      {412, "ConditionNotMet"},
      {409, "BlobAlreadyExists"},
  };

  for (const auto& c : cases) {
    auto code = ClassifyAzureError(c.http_status, c.error_code, /*transport_failure=*/false);
    ASSERT_TRUE(code.has_value()) << c.http_status << " " << c.error_code;
    EXPECT_EQ(*code, ExtendStatusCode::StoragePreConditionFailed) << c.http_status;
    EXPECT_EQ(CategoryForExtendStatusCode(*code), ErrorCategory::Conflict) << c.http_status;
    EXPECT_FALSE(RetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), milvus::StorageError) << c.http_status;
  }
}

// Anything not positively identified stays untagged and lands in the
// conservative bucket. Never invent retriability.
TEST(AzureErrorClassification, UnidentifiedStatusesStayUntagged) {
  const int unidentified[] = {400, 405, 411, 413, 416, 501, 507};
  for (int http_status : unidentified) {
    EXPECT_FALSE(ClassifyAzureError(http_status, "", /*transport_failure=*/false).has_value()) << http_status;
    EXPECT_EQ(SegcoreCodeFor(http_status), milvus::StorageError) << http_status;
  }

  // 409 that is not "already exists" is a different condition; not guessed at.
  EXPECT_FALSE(ClassifyAzureError(409, "LeaseIdMissing", /*transport_failure=*/false).has_value());

  // 412 gets the same treatment, which it did not used to. Azure answers 412
  // both for a genuine etag mismatch and for lease problems; classifying every
  // 412 as a precondition conflict was a guess, and it sat directly above the
  // 409 case that already refused to guess.
  EXPECT_FALSE(ClassifyAzureError(412, "LeaseIdMismatchWithBlobOperation", /*transport_failure=*/false).has_value());
  EXPECT_FALSE(ClassifyAzureError(412, "LeaseNotPresentWithBlobOperation", /*transport_failure=*/false).has_value());
  EXPECT_FALSE(ClassifyAzureError(412, "", /*transport_failure=*/false).has_value());
  EXPECT_EQ(SegcoreCodeFor(412, "LeaseIdMismatchWithBlobOperation"), milvus::StorageError);
  EXPECT_EQ(SegcoreCodeFor(409, "LeaseIdMissing"), milvus::StorageError);
}

// The two conditions the issue this fixes called out by name, stated as the
// end-to-end verdict a consumer sees.
TEST(AzureErrorClassification, ClosesTheTwoGapsFrom595) {
  // (a) A throttled/unavailable Azure account is retriable instead of being
  //     reported as a permanent storage error.
  EXPECT_EQ(SegcoreCodeFor(429, "TooManyRequests"), milvus::StorageTransientError);
  EXPECT_EQ(SegcoreCodeFor(503, "ServerBusy"), milvus::StorageTransientError);

  // (b) A missing blob is distinguishable from a generic storage failure.
  EXPECT_EQ(SegcoreCodeFor(404, "BlobNotFound"), milvus::ObjectNotExist);
  EXPECT_NE(SegcoreCodeFor(404, "BlobNotFound"), SegcoreCodeFor(400, ""));
}

// A missing container is a deployment mistake, a missing blob may be a GC race.
// Azure answers 404 to both; before this split the consumer was told to re-read
// its metadata in a case where no metadata could ever produce a container.
TEST(AzureErrorClassification, MissingContainerAndBlobAreDistinctSystemCodes) {
  // Azure reports it either way -- ErrorCode when it has one, ReasonPhrase when
  // it does not. Both must reach the same verdict, or the classification would
  // depend on which form the service happened to send.
  const std::vector<std::pair<std::string_view, std::string_view>> container_gone = {
      {"ContainerNotFound", ""},
      {"", "The specified container does not exist."},
      {"", "The specified filesystem does not exist."},
  };
  for (const auto& [error_code, reason] : container_gone) {
    auto code = ClassifyAzureError(404, error_code, /*transport_failure=*/false, reason);
    ASSERT_TRUE(code.has_value()) << error_code << "/" << reason;
    EXPECT_EQ(*code, ExtendStatusCode::StorageBucketNotFound) << error_code << "/" << reason;
    EXPECT_EQ(CategoryForExtendStatusCode(*code), ErrorCategory::System) << error_code << "/" << reason;
    EXPECT_EQ(SegcoreCodeFor(404, error_code, reason), milvus::BucketInvalid) << error_code << "/" << reason;
  }

  // Container-level operations retain their provenance even when a proxy
  // strips both Azure diagnostic strings from a bodyless 404.
  auto bodyless_container = ClassifyAzureError(404, "", /*transport_failure=*/false, "", AzureResourceKind::Container);
  ASSERT_TRUE(bodyless_container.has_value());
  EXPECT_EQ(*bodyless_container, ExtendStatusCode::StorageBucketNotFound);
  EXPECT_EQ(CategoryForExtendStatusCode(*bodyless_container), ErrorCategory::System);

  // A missing blob keeps the object-not-found code. This is the half that must
  // NOT move to BucketInvalid, which would page an operator for what may be a GC
  // race the consumer can resolve itself.
  auto blob = ClassifyAzureError(404, "BlobNotFound", /*transport_failure=*/false, "");
  ASSERT_TRUE(blob.has_value());
  EXPECT_EQ(*blob, ExtendStatusCode::StorageNotFound);
  EXPECT_EQ(CategoryForExtendStatusCode(*blob), ErrorCategory::System);
  EXPECT_EQ(SegcoreCodeFor(404, "BlobNotFound"), milvus::ObjectNotExist);
}

TEST(AzureErrorClassification, SyntheticBrokerUnexpectedStaysUnexpected) {
  Azure::Core::RequestFailedException error("broker failure");
  error.StatusCode = Azure::Core::Http::HttpStatusCode::InternalServerError;
  error.ErrorCode = kSyntheticBrokerUnexpectedErrorCode;

  auto status = AzureExceptionToStatus(error, "FetchSasToken:");

  EXPECT_TRUE(status.IsUnknownError()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

TEST(AzureErrorClassification, SyntheticBrokerOutOfMemoryStaysOutOfMemory) {
  Azure::Core::RequestFailedException error("broker allocation failure");
  error.StatusCode = Azure::Core::Http::HttpStatusCode::InternalServerError;
  error.ErrorCode = kSyntheticBrokerOutOfMemoryErrorCode;

  auto status = AzureExceptionToStatus(error, "FetchSasToken:");

  EXPECT_TRUE(status.IsOutOfMemory()) << status.ToString();
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::MemAllocateFailed);
}

// The producer/consumer round-trip: every LoonBroker* code the SAS policy
// stamps must map back to the exact ExtendStatusCode it started from. The
// prefix is what lets ClassifyAzureError short-circuit before the HTTP-status
// switch, so the synthetic status is deliberately varied -- and, for network,
// 503 rather than a "connection dropped" status HTTP does not have.
TEST(AzureErrorClassification, BrokerErrorCodeRoundTripsToExtendStatus) {
  struct Case {
    std::string_view error_code;
    ExtendStatusCode expected;
  };
  const Case cases[] = {
      {kSyntheticBrokerAccessDeniedErrorCode, ExtendStatusCode::StorageAccessDenied},
      {kSyntheticBrokerTimeoutErrorCode, ExtendStatusCode::StorageTransientTimeout},
      {kSyntheticBrokerThrottlingErrorCode, ExtendStatusCode::StorageTransientThrottling},
      {kSyntheticBrokerServiceUnavailableErrorCode, ExtendStatusCode::StorageTransientService},
      {kSyntheticBrokerNetworkErrorCode, ExtendStatusCode::StorageTransientNetwork},
  };
  for (const auto& c : cases) {
    auto code = BrokerErrorCodeToExtendStatus(c.error_code);
    ASSERT_TRUE(code.has_value()) << c.error_code;
    EXPECT_EQ(*code, c.expected) << c.error_code;
  }

  // The OOM and unexpected codes are not ExtendStatusCode conditions and stay
  // unmapped here -- AzureExceptionToStatus handles them before the classifier.
  EXPECT_FALSE(BrokerErrorCodeToExtendStatus(kSyntheticBrokerOutOfMemoryErrorCode).has_value());
  EXPECT_FALSE(BrokerErrorCodeToExtendStatus(kSyntheticBrokerUnexpectedErrorCode).has_value());

  // The prefix short-circuits the status switch, so the synthetic HTTP status
  // never leaks into the classification -- including the OOM/unexpected codes,
  // which must not fall through to 500 -> StorageTransientService.
  auto via_classifier = ClassifyAzureError(503, kSyntheticBrokerNetworkErrorCode, false);
  ASSERT_TRUE(via_classifier.has_value());
  EXPECT_EQ(*via_classifier, ExtendStatusCode::StorageTransientNetwork);

  EXPECT_FALSE(ClassifyAzureError(500, kSyntheticBrokerOutOfMemoryErrorCode, false).has_value());
  EXPECT_FALSE(ClassifyAzureError(500, kSyntheticBrokerUnexpectedErrorCode, false).has_value());
}

}  // namespace milvus_storage::fs::internal
