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

#include <optional>
#include <string_view>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/azure/azurefs_internal.h"

namespace milvus_storage::fs::internal {
namespace {

// nullopt means the classifier declined to tag it, which is what the caller
// actually produces: a plain IOError carrying no classification at all.
std::optional<ErrorCategory> CategoryFor(int http_status,
                                         std::string_view error_code = "",
                                         std::string_view reason_phrase = "") {
  auto code = ClassifyAzureError(http_status, error_code, /*transport_failure=*/false, reason_phrase);
  if (!code.has_value()) {
    return std::nullopt;
  }
  return CategoryForExtendStatusCode(*code);
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
    EXPECT_TRUE((CategoryForExtendStatusCode(*code) == ErrorCategory::Transient)) << c.http_status;
    EXPECT_EQ(CategoryForExtendStatusCode(*code), ErrorCategory::Transient) << c.http_status;
  }

  // A transport failure carries no status at all; it is identified by the
  // exception's dynamic type, not by status == 0.
  auto transport = ClassifyAzureError(0, "", /*transport_failure=*/true);
  ASSERT_TRUE(transport.has_value());
  EXPECT_EQ(*transport, ExtendStatusCode::StorageTransientNetwork);
  EXPECT_TRUE((CategoryForExtendStatusCode(*transport) == ErrorCategory::Transient));
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
  EXPECT_FALSE(CategoryFor(0).has_value());

  // Same input, but genuinely a transport failure: retriable.
  auto transport = ClassifyAzureError(0, "", /*transport_failure=*/true);
  ASSERT_TRUE(transport.has_value());
  EXPECT_EQ(CategoryForExtendStatusCode(*transport), ErrorCategory::Transient);
}

TEST(AzureErrorClassification, NonRetriableStatusesNeverLookTransient) {
  struct Case {
    int http_status;
    std::string_view error_code;
    ExtendStatusCode expected;
  };
  const Case cases[] = {
      // Not-found is fine-grained: a consumer can tell "data missing" from a
      // generic storage failure, matching what the S3 path already reports.
      {404, "BlobNotFound", ExtendStatusCode::AwsErrorNotFound},
      // Config, not Permanent: the credentials are operator configuration, so
      // this has to reach whoever owns the deployment (2006) rather than be
      // filed as a generic storage failure (2044). Non-retriable either way.
      {401, "", ExtendStatusCode::AwsErrorAccessDenied},
      {403, "AuthenticationFailed", ExtendStatusCode::AwsErrorAccessDenied},
  };

  for (const auto& c : cases) {
    auto code = ClassifyAzureError(c.http_status, c.error_code, /*transport_failure=*/false);
    ASSERT_TRUE(code.has_value()) << c.http_status << " " << c.error_code;
    EXPECT_EQ(*code, c.expected) << c.http_status;
    EXPECT_FALSE((CategoryForExtendStatusCode(*code) == ErrorCategory::Transient)) << c.http_status;

    // A non-retriable failure must never look transient, or a consumer
    // retry-storms a request that can never succeed.
    EXPECT_NE(CategoryForExtendStatusCode(*code), ErrorCategory::Transient) << c.http_status;
  }
}

// 412 and 409-already-exists are Conflict, not permanent: someone else won a
// race, and a re-read-then-rebase CAN succeed where the failure itself cannot
// be repaired. What does NOT work is resending the same conditional request,
// which fails identically forever.
//
// So the category is separate from Permanent (there is a recovery, and a
// consumer that can rebase must be able to find it) and separate from Transient
// (the recovery is not a replay, so the generic retryable bit stays false and
// the segcore code stays out of 2045). This test pins both halves; getting only
// one of them right is how this drifted before.
TEST(AzureErrorClassification, PreconditionFailuresAreConflictNotPermanent) {
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
    EXPECT_EQ(*code, ExtendStatusCode::AwsErrorPreConditionFailed) << c.http_status;
    EXPECT_EQ(CategoryForExtendStatusCode(*code), ErrorCategory::Conflict) << c.http_status;
    // Reachable as Conflict (the rebase signal survives) but not retryable and
    // not 2045 (a generic consumer is never told to replay the lost race).
    EXPECT_FALSE((CategoryForExtendStatusCode(*code) == ErrorCategory::Transient)) << c.http_status;
    EXPECT_NE(CategoryForExtendStatusCode(*code), ErrorCategory::Transient) << c.http_status;
  }
}

// Anything not positively identified stays untagged and lands in the
// conservative bucket. Never invent retriability.
TEST(AzureErrorClassification, UnidentifiedStatusesStayUntagged) {
  const int unidentified[] = {400, 405, 411, 413, 416, 501, 507};
  for (int http_status : unidentified) {
    EXPECT_FALSE(ClassifyAzureError(http_status, "", /*transport_failure=*/false).has_value()) << http_status;
    EXPECT_FALSE(CategoryFor(http_status).has_value()) << http_status;
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
  EXPECT_FALSE(CategoryFor(412, "LeaseIdMismatchWithBlobOperation").has_value());
  EXPECT_FALSE(CategoryFor(409, "LeaseIdMissing").has_value());
}

// The two conditions the issue this fixes called out by name, stated as the
// end-to-end verdict a consumer sees.
TEST(AzureErrorClassification, ClosesTheTwoGapsFrom595) {
  // (a) A throttled/unavailable Azure account is retriable instead of being
  //     reported as a permanent storage error.
  EXPECT_EQ(CategoryFor(429, "TooManyRequests"), ErrorCategory::Transient);
  EXPECT_EQ(CategoryFor(503, "ServerBusy"), ErrorCategory::Transient);

  // (b) A missing blob is distinguishable from a generic storage failure.
  EXPECT_EQ(CategoryFor(404, "BlobNotFound"), ErrorCategory::Missing);
  EXPECT_NE(CategoryFor(404, "BlobNotFound"), CategoryFor(400, ""));
}

// A missing container is a deployment mistake, a missing blob may be a GC race.
// Azure answers 404 to both; before this split the consumer was told to re-read
// its metadata in a case where no metadata could ever produce a container.
TEST(AzureErrorClassification, MissingContainerIsConfigNotMissing) {
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
    EXPECT_EQ(*code, ExtendStatusCode::AwsErrorBucketNotFound) << error_code << "/" << reason;
    EXPECT_EQ(CategoryForExtendStatusCode(*code), ErrorCategory::Config) << error_code << "/" << reason;
    EXPECT_EQ(CategoryFor(404, error_code, reason), ErrorCategory::Config) << error_code << "/" << reason;
  }

  // A missing blob keeps the old verdict. This is the half that must NOT move:
  // upgrading it to Config would page an operator for what is routinely a GC
  // race the consumer resolves by itself.
  auto blob = ClassifyAzureError(404, "BlobNotFound", /*transport_failure=*/false, "");
  ASSERT_TRUE(blob.has_value());
  EXPECT_EQ(*blob, ExtendStatusCode::AwsErrorNotFound);
  EXPECT_EQ(CategoryForExtendStatusCode(*blob), ErrorCategory::Missing);
  EXPECT_EQ(CategoryFor(404, "BlobNotFound"), ErrorCategory::Missing);
}

}  // namespace milvus_storage::fs::internal
