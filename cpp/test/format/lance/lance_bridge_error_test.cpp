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

// Regression tests for the classified error channel of the Rust bridges:
// the Rust side embeds a marker code into the (string-only) cxx error, the
// shared decoder rebuilds a structured arrow::Status, and the segcore mapping
// turns it into the right ErrorCode. These tests pin both the decoder table
// and the end-to-end not-found path through a real lance open.

#include <cerrno>
#include <string>

#include <arrow/status.h>
#include <arrow/util/io_util.h>
#include <gtest/gtest.h>

#include "bridge_error.h"
#include "lance_bridge.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_internal/ffi_error_code.h"

namespace milvus_storage::bridge {
namespace {

constexpr const char* kMarker = "__LOON_RUST_BRIDGE_ERRCODE__=";

TEST(BridgeErrorTest, NotFoundCodeBecomesEnoentDetail) {
  auto status = MakeBridgeErrorStatus(std::string(kMarker) + "12; dataset was not found");
  ASSERT_TRUE(status.IsIOError());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);
  // Marker must be stripped from the user-visible message.
  EXPECT_EQ(status.message().find("__LOON_"), std::string::npos);
  // End of the chain: fine-grained ObjectNotExist, never a transient code.
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist);
}

TEST(BridgeErrorTest, TransientCodeBecomesRetryableExtendDetail) {
  auto status = MakeBridgeErrorStatus(std::string(kMarker) + "109; too much write contention");
  ASSERT_TRUE(status.IsIOError());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
  EXPECT_TRUE(detail->retryable());
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageTransientError);
}

TEST(BridgeErrorTest, BridgePrivateCodesMapToArrowStatusCodes) {
  auto corrupt = MakeBridgeErrorStatus(std::string(kMarker) + "1001; corrupt file");
  EXPECT_TRUE(corrupt.IsInvalid());
  EXPECT_EQ(ToSegcoreError(corrupt).get_error_code(), milvus::DataFormatBroken);

  auto not_supported = MakeBridgeErrorStatus(std::string(kMarker) + "1002; unsupported feature");
  EXPECT_TRUE(not_supported.IsNotImplemented());
}

TEST(BridgeErrorTest, UnmarkedMessageStaysConservativeIOError) {
  auto status = MakeBridgeErrorStatus("some opaque failure");
  ASSERT_TRUE(status.IsIOError());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  // Untagged -> conservative non-retriable StorageError, never invented
  // retriability.
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

TEST(BridgeErrorTest, UnknownMarkerCodeFallsBackToIOError) {
  auto status = MakeBridgeErrorStatus(std::string(kMarker) + "424242; from a future version");
  ASSERT_TRUE(status.IsIOError());
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(status), nullptr);
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageError);
}

TEST(BridgeErrorTest, TranslatePreservesClassificationAndAddsContext) {
  auto tagged = MakeBridgeErrorStatus(std::string(kMarker) + "109; throttled");
  auto translated = TranslateBridgeStatus("reading chunk", tagged);
  auto detail = ExtendStatusDetail::UnwrapStatus(translated);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
  EXPECT_NE(translated.message().find("reading chunk"), std::string::npos);

  // A status whose *message* still carries the marker (arrow FFI stream
  // stringification, the mid-scan case) is decoded too.
  auto raw = arrow::Status::IOError(std::string(kMarker) + "12; object vanished mid-scan");
  auto decoded = TranslateBridgeStatus("stream", raw);
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(decoded), ENOENT);
  EXPECT_EQ(decoded.message().find("__LOON_"), std::string::npos);
}

TEST(BridgeErrorTest, TranslateDecodesMarkerRegardlessOfStatusCode) {
  // Mid-scan stream errors surface as whatever code the arrow C-stream import
  // assigns (the exporter maps Rust errors to EINVAL => Invalid, not IOError)
  // while still carrying the marker. Discrimination must be on marker
  // presence: a marker inside an Invalid must decode to the tagged class.
  auto midscan = TranslateBridgeStatus("stream", arrow::Status::Invalid(std::string(kMarker) + "109; throttled"));
  auto detail = ExtendStatusDetail::UnwrapStatus(midscan);
  ASSERT_NE(detail, nullptr) << midscan.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
  EXPECT_TRUE(detail->retryable());
  EXPECT_EQ(midscan.message().find("__LOON_"), std::string::npos) << midscan.ToString();
  EXPECT_EQ(ToSegcoreError(midscan).get_error_code(), milvus::StorageTransientError);

  auto midscan_notfound = TranslateBridgeStatus("stream", arrow::Status::Invalid(std::string(kMarker) + "12; gone"));
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(midscan_notfound), ENOENT) << midscan_notfound.ToString();
}

TEST(BridgeErrorTest, TranslateDoesNotDowngradeNonIOErrorStatuses) {
  // Bridge errors only travel as IOError strings; statuses arrow itself
  // produced (Invalid / OutOfMemory from ImportChunkedArray etc.) must pass
  // through with their StatusCode intact -- re-decoding them would downgrade
  // OOM (retriable 2034) into StorageError (non-retriable 2044).
  auto invalid = TranslateBridgeStatus("ctx", arrow::Status::Invalid("bad schema"));
  EXPECT_TRUE(invalid.IsInvalid()) << invalid.ToString();

  auto oom = TranslateBridgeStatus("ctx", arrow::Status::OutOfMemory("alloc failed"));
  ASSERT_TRUE(oom.IsOutOfMemory()) << oom.ToString();
  EXPECT_EQ(ToSegcoreError(oom).get_error_code(), milvus::MemAllocateFailed);

  auto not_impl = TranslateBridgeStatus("ctx", arrow::Status::NotImplemented("nope"));
  EXPECT_TRUE(not_impl.IsNotImplemented()) << not_impl.ToString();
}

// End-to-end: a real lance open against a nonexistent local dataset must come
// back as a classified not-found (ENOENT detail -> ObjectNotExist), not as an
// exception and not as an opaque IOError.
TEST(LanceBridgeErrorTest, OpenNonexistentDatasetClassifiesNotFound) {
  auto result = milvus_storage::lance::BlockingDataset::Open("/nonexistent-milvus-storage-test/lance-dataset");
  ASSERT_FALSE(result.ok());
  const auto& status = result.status();
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT) << status.ToString();
  EXPECT_EQ(status.message().find("__LOON_"), std::string::npos) << status.ToString();
  EXPECT_EQ(milvus_storage::ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist) << status.ToString();
}

}  // namespace
}  // namespace milvus_storage::bridge
