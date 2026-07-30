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

#include "common/EasyAssert.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/azure/azurefs_internal.h"

namespace milvus_storage::fs::internal {
namespace {

milvus::ErrorCode SegcoreCodeFor(int http_status, std::string_view error_code = "") {
  auto code = ClassifyAzureError(http_status, error_code, /*transport_failure=*/false);
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
    EXPECT_TRUE(DefaultRetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), milvus::StorageTransientError) << c.http_status;
  }

  // A transport failure carries no status at all; it is identified by the
  // exception's dynamic type, not by status == 0.
  auto transport = ClassifyAzureError(0, "", /*transport_failure=*/true);
  ASSERT_TRUE(transport.has_value());
  EXPECT_EQ(*transport, ExtendStatusCode::StorageTransientNetwork);
  EXPECT_TRUE(DefaultRetryableForExtendStatusCode(*transport));
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
      {404, "BlobNotFound", ExtendStatusCode::AwsErrorNotFound, milvus::ObjectNotExist},
      // Config, not Permanent: the credentials are operator configuration, so
      // this has to reach whoever owns the deployment (2006) rather than be
      // filed as a generic storage failure (2044). Non-retriable either way.
      {401, "", ExtendStatusCode::AwsErrorAccessDenied, milvus::ConfigInvalid},
      {403, "AuthenticationFailed", ExtendStatusCode::AwsErrorAccessDenied, milvus::ConfigInvalid},
  };

  for (const auto& c : cases) {
    auto code = ClassifyAzureError(c.http_status, c.error_code, /*transport_failure=*/false);
    ASSERT_TRUE(code.has_value()) << c.http_status << " " << c.error_code;
    EXPECT_EQ(*code, c.expected) << c.http_status;
    EXPECT_FALSE(DefaultRetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), c.segcore) << c.http_status;
    // A non-retriable failure must never look transient, or a consumer
    // retry-storms a request that can never succeed.
    EXPECT_NE(ToSegcoreErrorCode(*code), milvus::StorageTransientError) << c.http_status;
  }
}

// 412 and 409-already-exists used to sit in the test above, asserted permanent
// and non-retriable. They are Conflict: someone else won a race, and unlike a
// permanent failure a retry CAN succeed -- but only a re-read-then-retry, not a
// resend of the same conditional request, which fails identically forever.
//
// That is why Conflict is its own category rather than part of Transient, and
// why these cases need their own test: the two properties a permanent failure
// has (never retry, never look transient) are exactly the two these do not.
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
    EXPECT_TRUE(DefaultRetryableForExtendStatusCode(*code)) << c.http_status;
    EXPECT_EQ(ToSegcoreErrorCode(*code), milvus::StorageTransientError) << c.http_status;
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

}  // namespace milvus_storage::fs::internal
