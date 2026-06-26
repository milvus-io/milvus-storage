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

#include <gtest/gtest.h>

#include <arrow/status.h>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/result.h"

namespace milvus_storage::test {

// ArrowStatusToLoonCode routes an arrow failure onto a finer LOON code so the
// consumer can tell a transient failure (retriable) from a permanent one.
TEST(ArrowStatusToLoonCode, RoutesByCategory) {
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::OutOfMemory("oom")), LOON_MEMORY_ERROR);
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::IOError("io")), LOON_IO_ERROR);
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::Invalid("invalid")), LOON_DATA_ERROR);
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::TypeError("type")), LOON_DATA_ERROR);
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::KeyError("key")), LOON_DATA_ERROR);
  // Anything not specifically categorized stays the generic arrow error.
  EXPECT_EQ(ArrowStatusToLoonCode(arrow::Status::UnknownError("other")), LOON_ARROW_ERROR);
}

// The new codes must have matching strings (the static_assert in result_c.cpp
// also guards that error_strings stays in sync with LOON_ERRORCODE_MAX).
TEST(ErrorToString, CoversNewCodes) {
  EXPECT_EQ(error_to_string(LOON_IO_ERROR), "IO error");
  EXPECT_EQ(error_to_string(LOON_DATA_ERROR), "Data error");
}

}  // namespace milvus_storage::test
