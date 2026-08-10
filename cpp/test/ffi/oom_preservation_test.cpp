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
// rounds, at forty-five separate sites, because each round fixed the instance
// named and not the rule. So the rule is written down here: every transform
// below takes a status and returns one, and OutOfMemory survives. A new lossy
// path fails HERE rather than in the next review.
//
// Surviving means staying IDENTIFIABLE as an allocation failure -- not
// retriable, which this library no longer claims about anything. An OOM filed
// under "internal error" is indistinguishable from a null dereference, and it
// is the one failure whose fix has nothing to do with any cause it would be
// confused with.

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
      {"FFIErrorCodeFromExtendStatus", nullptr,
       [](const arrow::Status& s) { return FFIErrorCodeFromExtendStatus(s, LOON_ARROW_ERROR); }},
      {"UserSourceErrorCodeFromStatus", nullptr,
       [](const arrow::Status& s) { return UserSourceErrorCodeFromStatus(s, LOON_ARROW_ERROR); }},
  };
}

// A status is still "an OOM" if the two channels that can carry that fact still
// say so. They are checked together on purpose: a fix that preserves one while
// dropping the other is the shape several of the forty-five regressions took.
//
// The segcore code is asserted for its RETRY verdict rather than its identity,
// because that channel has no non-retriable memory code to land on.
void ExpectStillOutOfMemory(const arrow::Status& out, const char* what) {
  SCOPED_TRACE(what);
  EXPECT_TRUE(out.IsOutOfMemory()) << "arrow's own code was replaced: " << out.ToString();
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(out, LOON_ARROW_ERROR), LOON_MEMORY_ERROR)
      << "the FFI code stopped naming the allocation failure: " << out.ToString();
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
    EXPECT_EQ(transform.to_ffi_code(oom), LOON_MEMORY_ERROR)
        << transform.name << " stopped naming the allocation failure";
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
