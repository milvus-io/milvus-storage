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

#ifdef WITH_CRT

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>

#include "milvus-storage/filesystem/s3/s3_global.h"
#include "test_env.h"

namespace milvus_storage::test {

TEST(S3CrtBuildSupportTest, HeadersAndStaticClientSymbolsAreAvailable) {
  static_assert(std::is_class_v<Aws::S3Crt::S3CrtClient>);
  static_assert(std::is_default_constructible_v<Aws::S3Crt::S3CrtClientConfiguration>);

  ASSERT_STATUS_OK(EnsureS3Initialized());

  Aws::S3Crt::S3CrtClientConfiguration config;
  EXPECT_FALSE(config.region.empty());

  const char* service_name = Aws::S3Crt::S3CrtClient::GetServiceName();
  ASSERT_NE(service_name, nullptr);
  EXPECT_FALSE(std::string(service_name).empty());
}

}  // namespace milvus_storage::test

#endif  // WITH_CRT
