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

#include "milvus-storage/filesystem/metrics/error_classify.h"

namespace milvus_storage::test {

struct HttpCase {
  int code;
  OpStatus expected;
};

TEST(ErrorClassifyTest, HttpStatusMapping) {
  const HttpCase cases[] = {
      {404, OpStatus::NotFound},    {403, OpStatus::Auth},        {429, OpStatus::Throttled},
      {503, OpStatus::Throttled},   {408, OpStatus::Timeout},     {504, OpStatus::Timeout},
      {500, OpStatus::ServerError}, {400, OpStatus::ClientError}, {412, OpStatus::ClientError},
      {200, OpStatus::Unknown},
  };
  for (const auto& c : cases) {
    EXPECT_EQ(ClassifyHttpStatus(c.code), c.expected) << c.code;
  }
}

TEST(ErrorClassifyTest, ArrowStatusMapping) {
  EXPECT_EQ(ClassifyArrowStatus(arrow::Status::OK()), OpStatus::Ok);
  EXPECT_EQ(ClassifyArrowStatus(arrow::Status::IOError("boom")), OpStatus::Unknown);
  EXPECT_EQ(ClassifyArrowStatus(arrow::Status::Invalid("bad")), OpStatus::ClientError);
}

TEST(ErrorClassifyTest, S3Mapping) {
  EXPECT_EQ(ClassifyS3(503, /*throttle=*/true, false, false), OpStatus::Throttled);
  EXPECT_EQ(ClassifyS3(0, false, /*timeout=*/true, false), OpStatus::Timeout);
  EXPECT_EQ(ClassifyS3(0, false, false, /*connect=*/true), OpStatus::Network);
  EXPECT_EQ(ClassifyS3(404, false, false, false), OpStatus::NotFound);
  // Precedence: throttle wins over timeout/connect/http.
  EXPECT_EQ(ClassifyS3(404, true, true, true), OpStatus::Throttled);
}

}  // namespace milvus_storage::test
