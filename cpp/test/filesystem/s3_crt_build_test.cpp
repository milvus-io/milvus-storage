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

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "test_env.h"

namespace milvus_storage::test {

TEST(S3CrtBuildSupportTest, HeadersAndStaticClientSymbolsAreAvailable) {
  static_assert(std::is_class_v<Aws::S3Crt::S3CrtClient>);
  static_assert(std::is_default_constructible_v<Aws::S3Crt::S3CrtClientConfiguration>);

  ASSERT_STATUS_OK(EnsureS3Initialized());

  const char* service_name = Aws::S3Crt::S3CrtClient::GetServiceName();
  ASSERT_NE(service_name, nullptr);
  EXPECT_FALSE(std::string(service_name).empty());
}

TEST(S3CrtBuildSupportTest, OpenInputFileUsesCrtBackedAsyncFileWhenCrtEnabled) {
  if (!IsCloudEnv()) {
    GTEST_SKIP() << "CRT OpenInputFile smoke test skipped in non-cloud environment";
  }
  auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
  if (provider.ok() && provider.ValueOrDie() != kCloudProviderAWS) {
    GTEST_SKIP() << "CRT OpenInputFile smoke test only runs for AWS provider";
  }

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  ASSERT_AND_ASSIGN(auto fs, GetFileSystem(properties));

  const std::string base_path = GetTestBasePath("s3-crt-open-input-file-smoke");
  ASSERT_STATUS_OK(DeleteTestDir(fs, base_path));
  ASSERT_STATUS_OK(CreateTestDir(fs, base_path));

  const std::string object_path = base_path + "/crt-input-file.txt";
  const std::string data = "abcdefghi";
  ASSERT_AND_ASSIGN(auto output_stream, fs->OpenOutputStream(object_path));
  ASSERT_STATUS_OK(output_stream->Write(data.data(), static_cast<int64_t>(data.size())));
  ASSERT_STATUS_OK(output_stream->Close());

  ASSERT_AND_ASSIGN(auto input_file, fs->OpenInputFile(object_path));
  ASSERT_NE(dynamic_cast<milvus_storage::NonBlockingReadAtFile*>(input_file.get()), nullptr);

  auto async_result = input_file->ReadAsync({}, 2, 4).result();
  ASSERT_STATUS_OK(async_result.status());
  ASSERT_EQ(async_result.ValueOrDie()->ToString(), "cdef");

  ASSERT_AND_ASSIGN(auto sync_buffer, input_file->ReadAt(0, 3));
  ASSERT_EQ(sync_buffer->ToString(), "abc");
  ASSERT_STATUS_OK(input_file->Close());

  ASSERT_STATUS_OK(DeleteTestDir(fs, base_path));
}

}  // namespace milvus_storage::test

#endif  // WITH_CRT
