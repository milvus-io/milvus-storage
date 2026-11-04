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

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include <aws/core/auth/STSCredentialsProvider.h>

#include "milvus-storage/filesystem/s3/s3_fs.h"
#include "test_env.h"

namespace milvus_storage {
namespace {

class ScopedEnvVar {
  public:
  ScopedEnvVar(const std::string& name, const std::string& value) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    setenv(name.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// Unsets an env var for the scope lifetime, restoring it on destruction.
class ScopedEnvUnset {
  public:
  explicit ScopedEnvUnset(const std::string& name) : name_(name) {
    const char* old = std::getenv(name.c_str());
    if (old) {
      had_value_ = true;
      old_value_ = old;
    }
    unsetenv(name.c_str());
  }

  ~ScopedEnvUnset() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvUnset(const ScopedEnvUnset&) = delete;
  ScopedEnvUnset& operator=(const ScopedEnvUnset&) = delete;

  private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

// RAII helper for temporary files
class TempFile {
  public:
  explicit TempFile(const std::string& content) {
    path_ = "/tmp/test_oidc_token_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    std::ofstream ofs(path_);
    ofs << content;
    ofs.close();
  }

  ~TempFile() { std::remove(path_.c_str()); }

  [[nodiscard]] const std::string& path() const { return path_; }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  private:
  std::string path_;
};

class S3ProviderTest : public ::testing::Test {
  protected:
  static void SetUpTestSuite() {
    const auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != "aws") {
      return;
    }
    // Share 2.6's initialization guard and cleanup with the existing S3 tests.
    ArrowFileSystemConfig config;
    config.cloud_provider = "aws";
    S3FileSystemProducer producer(config);
    producer.InitS3();
  }

  void SetUp() override {
    const auto provider = GetEnvVar(ENV_VAR_CLOUD_PROVIDER);
    if (provider.ok() && provider.ValueOrDie() != "aws") {
      GTEST_SKIP() << "S3 provider tests only run for AWS provider";
    }
  }
};

TEST_F(S3ProviderTest, AwsIamCredentialsPreferEnvironmentThenWebIdentityThenInstanceProfile) {
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "ENV_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "ENV_SK");
  ScopedEnvUnset unset_session_token("AWS_SESSION_TOKEN");
  ScopedEnvVar set_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvVar enable_instance_profile("AWS_EC2_METADATA_ENABLE", "TrUe");
  ScopedEnvVar enable_imds("AWS_EC2_METADATA_DISABLED", "false");
  ScopedEnvUnset unset_container_uri("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset unset_container_full_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");

  ArrowFileSystemConfig config;
  config.cloud_provider = "aws";
  config.use_iam = true;

  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  const auto credentials = options.credentials_provider->GetAWSCredentials();
  EXPECT_EQ(credentials.GetAWSAccessKeyId(), "ENV_AK");
  EXPECT_EQ(credentials.GetAWSSecretKey(), "ENV_SK");
  EXPECT_TRUE(credentials.GetSessionToken().empty());

  // Keep EC2 instance credentials behind environment credentials and Web Identity.
  const auto chain = std::dynamic_pointer_cast<AwsCredentialsProviderChain>(options.credentials_provider);
  ASSERT_NE(chain, nullptr);
  const auto& providers = chain->GetProviders();
  ASSERT_EQ(providers.size(), 3u);
  EXPECT_NE(std::dynamic_pointer_cast<Aws::Auth::EnvironmentAWSCredentialsProvider>(providers[0]), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<Aws::Auth::STSAssumeRoleWebIdentityCredentialsProvider>(providers[1]), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<Aws::Auth::InstanceProfileCredentialsProvider>(providers[2]), nullptr);

  // External callers can construct the same policy through the public header.
  const auto external_chain = std::make_shared<AwsCredentialsProviderChain>();
  EXPECT_EQ(external_chain->GetAWSCredentials().GetAWSAccessKeyId(), "ENV_AK");
  EXPECT_TRUE(external_chain->GetAWSCredentials().GetSessionToken().empty());
}

TEST_F(S3ProviderTest, AwsIamCredentialsExcludeInstanceProfileUnlessEnabled) {
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "ENV_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "ENV_SK");
  ScopedEnvVar set_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvVar enable_imds("AWS_EC2_METADATA_DISABLED", "false");
  ScopedEnvUnset restore_enable("AWS_EC2_METADATA_ENABLE");

  ArrowFileSystemConfig config;
  config.cloud_provider = "aws";
  config.use_iam = true;
  S3FileSystemProducer producer(config);

  const char* const disabled_values[] = {nullptr, "", "false", "1"};
  for (const auto* const value : disabled_values) {
    SCOPED_TRACE(value == nullptr ? "unset" : value);
    if (value == nullptr) {
      ASSERT_EQ(unsetenv("AWS_EC2_METADATA_ENABLE"), 0);
    } else {
      ASSERT_EQ(setenv("AWS_EC2_METADATA_ENABLE", value, 1), 0);
    }
    ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
    EXPECT_EQ(options.credentials_provider->GetAWSCredentials().GetAWSAccessKeyId(), "ENV_AK");
    const auto chain = std::dynamic_pointer_cast<Aws::Auth::AWSCredentialsProviderChain>(options.credentials_provider);
    ASSERT_NE(chain, nullptr);
    const auto& providers = chain->GetProviders();
    ASSERT_EQ(providers.size(), 2u);
    EXPECT_NE(std::dynamic_pointer_cast<Aws::Auth::EnvironmentAWSCredentialsProvider>(providers[0]), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<Aws::Auth::STSAssumeRoleWebIdentityCredentialsProvider>(providers[1]), nullptr);
  }
}

TEST_F(S3ProviderTest, AwsIamCredentialsDoNotFallBackToProfileAfterEnvironmentCredentialsDisappear) {
  TempFile credentials_file(
      "[milvus-storage-credential-chain-test]\n"
      "aws_access_key_id = PROFILE_AK\n"
      "aws_secret_access_key = PROFILE_SK\n");
  TempFile config_file("");
  ScopedEnvVar set_credentials_file("AWS_SHARED_CREDENTIALS_FILE", credentials_file.path());
  ScopedEnvVar set_config_file("AWS_CONFIG_FILE", config_file.path());
  ScopedEnvVar set_profile("AWS_PROFILE", "milvus-storage-credential-chain-test");
  ScopedEnvVar set_default_profile("AWS_DEFAULT_PROFILE", "milvus-storage-credential-chain-test");
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "ENV_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "ENV_SK");
  ScopedEnvVar set_session_token("AWS_SESSION_TOKEN", "ENV_TOKEN");
  ScopedEnvVar set_region("AWS_DEFAULT_REGION", "us-east-1");
  ScopedEnvUnset unset_role("AWS_ROLE_ARN");
  ScopedEnvUnset unset_token_file("AWS_WEB_IDENTITY_TOKEN_FILE");
  ScopedEnvVar enable_imds("AWS_EC2_METADATA_DISABLED", "false");
  ScopedEnvUnset disable_instance_profile("AWS_EC2_METADATA_ENABLE");
  ScopedEnvUnset unset_container_uri("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
  ScopedEnvUnset unset_container_full_uri("AWS_CONTAINER_CREDENTIALS_FULL_URI");

  ArrowFileSystemConfig config;
  config.cloud_provider = "aws";
  config.use_iam = true;

  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  {
    ScopedEnvUnset unset_access_key("AWS_ACCESS_KEY_ID");
    ScopedEnvUnset unset_secret_key("AWS_SECRET_ACCESS_KEY");
    ScopedEnvUnset unset_session_token("AWS_SESSION_TOKEN");
    EXPECT_TRUE(options.credentials_provider->GetAWSCredentials().IsEmpty());
  }
  // A failed lookup must not pin the chain to another identity after recovery.
  EXPECT_EQ(options.credentials_provider->GetAWSCredentials().GetAWSAccessKeyId(), "ENV_AK");
}

TEST_F(S3ProviderTest, AwsExplicitCredentialsTakePrecedenceOverEnvironment) {
  ScopedEnvVar set_access_key("AWS_ACCESS_KEY_ID", "ENV_AK");
  ScopedEnvVar set_secret_key("AWS_SECRET_ACCESS_KEY", "ENV_SK");

  ArrowFileSystemConfig config;
  config.cloud_provider = "aws";
  config.use_iam = false;
  config.access_key_id = "CONFIG_AK";
  config.access_key_value = "CONFIG_SK";

  S3FileSystemProducer producer(config);
  ASSERT_AND_ASSIGN(auto options, producer.CreateS3Options());
  const auto credentials = options.credentials_provider->GetAWSCredentials();
  EXPECT_EQ(credentials.GetAWSAccessKeyId(), "CONFIG_AK");
  EXPECT_EQ(credentials.GetAWSSecretKey(), "CONFIG_SK");
}

}  // namespace
}  // namespace milvus_storage
