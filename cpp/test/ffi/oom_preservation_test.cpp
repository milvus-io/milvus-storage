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

// One invariant, enumerated: a status that says OutOfMemory must still say so
// after crossing anything in this library.
//
// This file exists because the alternative did not work. "OOM loses its
// classification" was reported and fixed eleven times across thirteen review
// rounds, at forty-five separate sites -- the FFI catch-all, WrapExtendError,
// the segment reader and writer, packed reader and writer, manifest
// serialization, the AWS core-error path, the S3 batch-delete aggregate, the
// vortex bridge's context wrapper, both FFI exports, the async vortex
// callback. Each round I fixed the site that was named and the next round
// found the next one, because I was patching instances of a rule nobody had
// written down.
//
// So it is written down here. Every transform below takes a status and returns
// one; the property is that OutOfMemory survives. Adding a new transform means
// adding a row, and a new lossy path fails HERE rather than in the next
// review.
//
// Why it matters more than the message: milvus's merr treats the retriable
// codes as retriable, and OOM is the condition most likely to succeed on a
// second attempt -- another node, or this one a moment later, has the memory.
// Every one of those forty-five sites turned it into something permanent,
// which is the one answer guaranteed to be wrong.

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include <arrow/status.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_internal/result.h"
#include "vortex_bridge.h"

namespace milvus_storage {
namespace {

struct Transform {
  const char* name;
  // What the code does to a status on its way somewhere else.
  std::function<arrow::Status(const arrow::Status&)> apply;
  // Some transforms answer in the FFI code space rather than by returning a
  // status; those set this instead.
  std::function<int(const arrow::Status&)> to_ffi_code = nullptr;
};

std::vector<Transform> AllTransforms() {
  return {
      {"WrapExtendError(PackedUnexpected, ...)",
       [](const arrow::Status& s) { return WrapExtendError(ExtendStatusCode::PackedUnexpected, "wrapped", s); }},
      {"WrapExtendError(ManifestCorrupted, ...)",
       [](const arrow::Status& s) { return WrapExtendError(ExtendStatusCode::ManifestCorrupted, "wrapped", s); }},
      {"MakeVortexErrorStatus(context, status)",
       [](const arrow::Status& s) { return vortex::MakeVortexErrorStatus("reading vortex", s); }},
      {"ToSegcoreError", nullptr,
       [](const arrow::Status& s) { return static_cast<int>(ToSegcoreError(s).get_error_code()); }},
      {"FFIErrorCodeFromExtendStatus", nullptr,
       [](const arrow::Status& s) { return FFIErrorCodeFromExtendStatus(s, LOON_ARROW_ERROR); }},
      {"UserSourceErrorCodeFromStatus", nullptr,
       [](const arrow::Status& s) { return UserSourceErrorCodeFromStatus(s, LOON_ARROW_ERROR); }},
  };
}

// A status is still "an OOM" if any of the three channels this library uses to
// carry that fact still says so. They are checked together on purpose: a fix
// that preserves one while dropping the others is the shape several of the
// forty-five regressions took.
void ExpectStillOutOfMemory(const arrow::Status& out, const char* what) {
  SCOPED_TRACE(what);
  EXPECT_TRUE(out.IsOutOfMemory()) << "arrow's own code was replaced: " << out.ToString();
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(out, LOON_ARROW_ERROR), LOON_MEMORY_ERROR)
      << "the FFI code stopped being retriable: " << out.ToString();
  EXPECT_EQ(ToSegcoreError(out).get_error_code(), milvus::MemAllocateFailed)
      << "segcore stopped seeing memory pressure: " << out.ToString();
}

}  // namespace

TEST(OomPreservationTest, EveryTransformKeepsOutOfMemory) {
  for (const auto& transform : AllTransforms()) {
    const auto oom = arrow::Status::OutOfMemory("allocation failed");
    if (transform.apply) {
      ExpectStillOutOfMemory(transform.apply(oom), transform.name);
      continue;
    }
    SCOPED_TRACE(transform.name);
    // Code-space transforms: the two retriable answers this library has.
    const int code = transform.to_ffi_code(oom);
    const bool retriable_answer = (code == LOON_MEMORY_ERROR) || (code == milvus::MemAllocateFailed);
    EXPECT_TRUE(retriable_answer) << transform.name << " answered " << code
                                  << ", which is not the memory-pressure code in its space";
  }
}

// The same invariant, one level down: whatever a transform does to an OOM, it
// must not make it non-retriable. Stated separately because this is the part
// that actually reaches milvus's retry policy, and a transform could in
// principle preserve "some classification" while losing retriability.
TEST(OomPreservationTest, NoTransformMakesOutOfMemoryPermanent) {
  for (const auto& transform : AllTransforms()) {
    if (!transform.apply) {
      continue;
    }
    SCOPED_TRACE(transform.name);
    auto out = transform.apply(arrow::Status::OutOfMemory("allocation failed"));
    const int code = FFIErrorCodeFromExtendStatus(out, LOON_ARROW_ERROR);
    EXPECT_TRUE(loon_ffi_is_retryable_errcode(code))
        << transform.name << " produced code " << code << " (" << loon_ffi_error_name(code)
        << "), which tells the caller not to retry an allocation failure";
  }
}

// The negative half. Without it the tests above are satisfiable by a transform
// that calls everything OutOfMemory, which would be a far worse bug -- it would
// put genuinely permanent failures into a retry loop forever.
TEST(OomPreservationTest, TransformsDoNotInventOutOfMemory) {
  const std::vector<arrow::Status> not_oom = {
      arrow::Status::IOError("plain io failure"),
      arrow::Status::Invalid("bad argument"),
      MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "denied", "denied"),
      MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "slow down", "slow down"),
      MakeExtendError(ExtendStatusCode::ManifestCorrupted, "bad bytes", "bad bytes"),
  };

  for (const auto& transform : AllTransforms()) {
    if (!transform.apply) {
      continue;
    }
    for (const auto& input : not_oom) {
      SCOPED_TRACE(std::string(transform.name) + " <- " + input.ToString());
      auto out = transform.apply(input);
      EXPECT_FALSE(out.IsOutOfMemory()) << "invented memory pressure that the input never claimed";
    }
  }
}

}  // namespace milvus_storage
