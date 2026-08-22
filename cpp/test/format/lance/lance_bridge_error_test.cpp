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
#include "milvus-storage/ffi_internal/result.h"

namespace milvus_storage::bridge {
namespace {

constexpr const char* kMarker = "__LOON_RUST_BRIDGE_ERRCODE__=";

// Code 12 is still decoded -- the vortex fork and the filesystem layer speak
// the errno channel -- even though no bridge here emits it any more.
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
  auto status = MakeBridgeErrorStatus(std::string(kMarker) + "109; throttled by the object store");
  ASSERT_TRUE(status.IsIOError());
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageTransientThrottling);
  EXPECT_TRUE(detail->retryable());
  EXPECT_EQ(ToSegcoreError(status).get_error_code(), milvus::StorageTransientError);
}

TEST(BridgeErrorTest, AccessDeniedMapsToExternalSourceInvalid) {
  auto status = MakeBridgeErrorStatus(std::string(kMarker) + "105; external source rejected credentials");
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr);
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageAccessDenied);
  EXPECT_EQ(ExternalSourceErrorCodeFromStatus(status, LOON_ARROW_ERROR), LOON_SOURCE_INVALID);
}

TEST(BridgeErrorTest, BridgePrivateCodesMapToArrowStatusCodes) {
  auto corrupt = MakeBridgeErrorStatus(std::string(kMarker) + "1001; corrupt file");
  auto corrupt_detail = ExtendStatusDetail::UnwrapStatus(corrupt);
  ASSERT_NE(corrupt_detail, nullptr);
  EXPECT_EQ(corrupt_detail->code(), ExtendStatusCode::DataCorrupted);
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

// A failure that did not come from a Rust bridge keeps whatever classification
// its own producer gave it. Arrow's C-stream importer raises Invalid /
// NotImplemented on its own, and rewriting those into a blanket IOError would
// throw away a fact nobody can recover downstream.
TEST(BridgeErrorTest, TranslateKeepsNonBridgeClassification) {
  auto invalid = TranslateBridgeStatus("reading chunk", arrow::Status::Invalid("stream schema is not importable"));
  EXPECT_TRUE(invalid.IsInvalid()) << invalid.ToString();
  EXPECT_NE(invalid.message().find("reading chunk"), std::string::npos);

  auto not_implemented =
      TranslateBridgeStatus("reading chunk", arrow::Status::NotImplemented("importing this type is unsupported"));
  EXPECT_TRUE(not_implemented.IsNotImplemented()) << not_implemented.ToString();

  // Translating twice must be a no-op on the classification, not a slow slide
  // down to IOError.
  auto decoded = TranslateBridgeStatus("stream", MakeBridgeErrorStatus(std::string(kMarker) + "1002; unsupported"));
  EXPECT_TRUE(TranslateBridgeStatus("outer", decoded).IsNotImplemented());
}

TEST(BridgeErrorTest, DecodeOnlyFiresOnTheMarker) {
  EXPECT_FALSE(DecodeBridgeErrorStatus("plain arrow failure").has_value());
  auto decoded = DecodeBridgeErrorStatus(std::string(kMarker) + "12; gone");
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(*decoded), ENOENT);
  EXPECT_EQ(StripBridgeErrorMarker(std::string(kMarker) + "12; gone"), "gone");
}

// The side channel is the primary channel now. This is what it buys: a bridge
// error whose MESSAGE contains marker-looking text cannot dictate its own
// classification when a real code was recorded -- the recorded code wins, and
// the message is only consulted when nothing was recorded. The vortex bridge
// has pinned the same property since it introduced the side channel.
TEST(BridgeErrorTest, SideChannelWinsOverMessageText) {
  milvus_storage::bridge::ClearBridgeErrorChannel();
  // Nothing recorded: the message is all we have, so the marker is honoured.
  auto from_message =
      milvus_storage::bridge::BridgeErrorStatusFromException((std::string(kMarker) + "109; throttled").c_str());
  auto message_detail = ExtendStatusDetail::UnwrapStatus(from_message);
  ASSERT_NE(message_detail, nullptr) << from_message.ToString();
  EXPECT_EQ(message_detail->code(), ExtendStatusCode::StorageTransientThrottling);

  // A recorded classification outranks whatever the text claims.
  internal::RecordBridgeErrorForTest(LOON_STORAGE_ACCESS_DENIED, "credentials rejected");
  auto from_channel = milvus_storage::bridge::BridgeErrorStatusFromException(
      (std::string(kMarker) + "109; a path that looks like a marker").c_str());
  auto channel_detail = ExtendStatusDetail::UnwrapStatus(from_channel);
  ASSERT_NE(channel_detail, nullptr) << from_channel.ToString();
  EXPECT_EQ(channel_detail->code(), ExtendStatusCode::StorageAccessDenied) << from_channel.ToString();
  EXPECT_FALSE(channel_detail->retryable());

  // Taking it empties the slot: a stale code cannot attach to the next failure.
  auto after = milvus_storage::bridge::BridgeErrorStatusFromException("plain failure");
  EXPECT_EQ(ExtendStatusDetail::UnwrapStatus(after), nullptr) << after.ToString();
}

// End-to-end: a real lance open against a nonexistent local dataset must come
// back as a classified not-found (ENOENT detail -> ObjectNotExist), not as an
// exception and not as an opaque IOError.
TEST(LanceBridgeErrorTest, OpenNonexistentDatasetClassifiesNotFound) {
  auto result = milvus_storage::lance::BlockingDataset::Open("/nonexistent-milvus-storage-test/lance-dataset");
  ASSERT_FALSE(result.ok());
  const auto& status = result.status();
  // Bridges report a missing object through the taxonomy channel, so that a
  // consumer has one place to look regardless of which bridge answered.
  auto detail = ExtendStatusDetail::UnwrapStatus(status);
  ASSERT_NE(detail, nullptr) << status.ToString();
  EXPECT_EQ(detail->code(), ExtendStatusCode::StorageNotFound) << status.ToString();
  EXPECT_FALSE(detail->retryable()) << status.ToString();
  EXPECT_EQ(status.message().find("__LOON_"), std::string::npos) << status.ToString();
  // Same landing as the errno channel: the split must be invisible downstream.
  EXPECT_EQ(milvus_storage::ToSegcoreError(status).get_error_code(), milvus::ObjectNotExist) << status.ToString();
}

}  // namespace
}  // namespace milvus_storage::bridge
