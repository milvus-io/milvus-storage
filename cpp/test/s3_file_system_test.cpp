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

#include <memory>
#include <type_traits>

#include "milvus-storage/filesystem/s3/multi_part_upload_s3_fs.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"

#include "test_util.h"

namespace milvus_storage {

class S3FsTest : public ::testing::Test {
  protected:
  void SetUp() override {
    std::string cloud_provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER).ValueOr("");
    std::string storage_type = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    std::string access_key_id = GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("");
    std::string access_key_value = GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("");
    std::string address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
    std::string bucket_name = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    std::string region = GetEnvVar(ENV_VAR_REGION).ValueOr("");

    if (cloud_provider.empty() || storage_type.empty() || address.empty() || bucket_name.empty()) {
      GTEST_SKIP() << "S3 credentials not set. Please set environment variables:\n"
                   << "CLOUD_PROVIDER, STORAGE_TYPE, ACCESS_KEY, SECRET_KEY, ADDRESS, BUCKET_NAME, REGION";
    }

    auto conf = ArrowFileSystemConfig();

    conf.cloud_provider = cloud_provider;
    conf.root_path = "/";
    conf.storage_type = storage_type;
    conf.request_timeout_ms = 1000;
    conf.use_ssl = false;
    conf.log_level = "debug";
    conf.region = region;
    conf.address = address;
    conf.bucket_name = bucket_name;
    conf.use_virtual_host = false;
    // not use iam
    if (!access_key_id.empty() && !access_key_value.empty()) {
      conf.use_iam = false;
      conf.access_key_id = access_key_id;
      conf.access_key_value = access_key_value;
    } else {
      conf.use_iam = true;
      conf.access_key_id = "";
      conf.access_key_value = "";
      // azure should provide access key
      if (conf.cloud_provider == "azure") {
        conf.access_key_id = access_key_id;
      }
    }

    ArrowFileSystemSingleton::GetInstance().Init(conf);
    fs_ = ArrowFileSystemSingleton::GetInstance().GetArrowFileSystem();
  }

  ArrowFileSystemPtr fs_;
};

class A {
  public:
  virtual ~A() = default;
};

class C {
  public:
  virtual ~C() = default;
};

class B : public A, public C {};
class D : public A {};

bool inheritsFromC(const std::shared_ptr<A>& ptr) { return std::dynamic_pointer_cast<C>(ptr) != nullptr; }

TEST_F(S3FsTest, TestExtend) {
  std::shared_ptr<A> b_ptr = std::make_shared<B>();
  std::shared_ptr<A> d_ptr = std::make_shared<D>();

  ASSERT_TRUE(inheritsFromC(b_ptr));
  ASSERT_FALSE(inheritsFromC(d_ptr));
}

TEST_F(S3FsTest, ConditionalWrite) {
  std::string bucket_name = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
  std::string file_to = bucket_name + "/test_conditional_write.txt";

  // Ensure source file does not exist
  (void)fs_->DeleteFile(file_to);

  std::string content1 = "This is a test file for conditional write.";
  std::string content2 = "This is a test file for conditional write 2.";

  ASSERT_TRUE(milvus_storage::ExtendFileSystem::IsExtendFileSystem(fs_));

  auto fs_ext = std::dynamic_pointer_cast<milvus_storage::ExtendFileSystem>(fs_);

  // Create source file
  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content1.c_str()), content1.size());

    ASSERT_AND_ASSIGN(auto output_stream, fs_ext->OpenConditionalOutputStream(file_to));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_OK(output_stream->Close());
  }

  {
    std::shared_ptr<arrow::Buffer> buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(content2.c_str()), content2.size());

    ASSERT_AND_ASSIGN(auto output_stream, fs_ext->OpenConditionalOutputStream(file_to));
    ASSERT_STATUS_OK(output_stream->Write(buffer));
    ASSERT_STATUS_NOT_OK(output_stream->Close());
  }
}

}  // namespace milvus_storage