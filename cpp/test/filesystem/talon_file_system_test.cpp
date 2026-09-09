// Copyright 2026 Zilliz
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

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#ifdef WITH_TALON
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/io_util.h>
#endif

#include "milvus-storage/common/config.h"
#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/fs.h"
#ifdef WITH_TALON
#include "milvus-storage/filesystem/talon/talon_file_system_producer.h"
#include "talon_bridge.h"
#endif
#include "milvus-storage/properties.h"
#include "milvus-storage/reader.h"
#include "milvus-storage/writer.h"
#include "test_env.h"

namespace milvus_storage::test {

namespace {

class ScopedEnv final {
  public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) {
      old_value_ = old;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~ScopedEnv() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  private:
  const std::string name_;
  std::optional<std::string> old_value_;
};

}  // namespace

#ifdef WITH_TALON
namespace {

class MetadataInputFile final : public arrow::io::BufferReader {
  public:
  MetadataInputFile() : arrow::io::BufferReader(arrow::Buffer::FromString("payload")) {}

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override { return metadata_result; }

  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    async_pool = io_context.pool();
    return async_result.is_valid()
               ? async_result
               : arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::MakeFinished(metadata_result);
  }

  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> metadata_result{
      std::shared_ptr<const arrow::KeyValueMetadata>(
          arrow::key_value_metadata({"ETag", "Content-Length", "owner"}, {"\"origin-etag\"", "7", "test-owner"}))};
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> async_result;
  arrow::MemoryPool* async_pool = nullptr;
};

class RejectingMemoryPool final : public arrow::ProxyMemoryPool {
  public:
  RejectingMemoryPool() : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    last_requested_size = size;
    if (size == 0) {
      return arrow::ProxyMemoryPool::Allocate(size, alignment, out);
    }
    // Stop before network I/O so tests can inspect the allocation request itself.
    return arrow::Status::OutOfMemory("Rejected test allocation");
  }

  int64_t last_requested_size = -1;
};

class RecordingFileSystem final : public arrow::fs::LocalFileSystem {
  public:
  explicit RecordingFileSystem(std::string identity,
                               const arrow::io::IOContext& io_context = arrow::io::default_io_context())
      : arrow::fs::LocalFileSystem(io_context), identity_(std::move(identity)) {}

  std::string type_name() const override { return "recording"; }

  bool Equals(const arrow::fs::FileSystem& other) const override {
    ++equals_calls_;
    if (this == &other) {
      return true;
    }
    const auto* recording = dynamic_cast<const RecordingFileSystem*>(&other);
    return recording != nullptr && identity_ == recording->identity_;
  }

  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& selector) override {
    ++synchronous_listing_calls_;
    return arrow::fs::FileInfoVector{
        arrow::fs::FileInfo(selector.base_dir + "/synchronous", arrow::fs::FileType::File)};
  }

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) override {
    ++generator_calls_;
    last_generator_selector_ = selector;
    std::vector<arrow::fs::FileInfoVector> batches;
    batches.push_back({arrow::fs::FileInfo(selector.base_dir + "/first", arrow::fs::FileType::File)});
    batches.push_back({arrow::fs::FileInfo(selector.base_dir + "/second", arrow::fs::FileType::File)});
    return arrow::MakeVectorGenerator(std::move(batches));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ++input_open_calls;
    last_input_path = path;
    ARROW_RETURN_NOT_OK(input_open_status);
    return input_file;
  }

  std::shared_ptr<MetadataInputFile> input_file = std::make_shared<MetadataInputFile>();
  arrow::Status input_open_status;
  int input_open_calls = 0;
  std::string last_input_path;

  int synchronous_listing_calls() const { return synchronous_listing_calls_; }
  int generator_calls() const { return generator_calls_; }
  int equals_calls() const { return equals_calls_; }
  const arrow::fs::FileSelector& last_generator_selector() const { return last_generator_selector_; }

  private:
  const std::string identity_;
  mutable int equals_calls_ = 0;
  int synchronous_listing_calls_ = 0;
  int generator_calls_ = 0;
  arrow::fs::FileSelector last_generator_selector_;
};

class TalonFileSystemServiceFreeTest : public ::testing::Test {
  protected:
  static ArrowFileSystemConfig Config() {
    ArrowFileSystemConfig config;
    config.storage_type = "remote";
    config.cloud_provider = kCloudProviderAWS;
    config.bucket_name = "test-bucket";
    config.talon_enabled = true;
    config.talon_coordinator = "127.0.0.1:7000";
    config.talon_block_size = 8388608;
    return config;
  }

  static arrow::Result<ArrowFileSystemPtr> Wrap(const ArrowFileSystemConfig& config,
                                                std::shared_ptr<RecordingFileSystem> origin) {
    ARROW_ASSIGN_OR_RAISE(auto talon_fs, TalonFileSystemProducer(config, std::move(origin)).Make());
    return std::make_shared<FileSystemProxy>(config.bucket_name, std::move(talon_fs));
  }
};

class TalonIntegrationTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsTalonEnv()) {
      GTEST_SKIP() << "Talon integration environment is not configured";
    }

    ASSERT_STATUS_OK(InitTestProperties(properties_));
    FilesystemCache::getInstance().clean();
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    path_ = "talon-integration/" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "-" +
            std::to_string(getpid());
    parquet_path_ = path_ + "-parquet";
    expected_.resize(1024 * 1024);
    for (size_t i = 0; i < expected_.size(); ++i) {
      expected_[i] = static_cast<uint8_t>(i % 251);
    }

    ASSERT_AND_ASSIGN(auto output, fs_->OpenOutputStream(path_));
    ASSERT_STATUS_OK(output->Write(expected_.data(), static_cast<int64_t>(expected_.size())));
    ASSERT_STATUS_OK(output->Close());
  }

  void TearDown() override {
    if (fs_ != nullptr) {
      (void)fs_->DeleteFile(path_);
      (void)fs_->DeleteDir(parquet_path_);
      FilesystemCache::getInstance().clean();
    }
  }

  api::Properties properties_;
  ArrowFileSystemPtr fs_;
  std::string path_;
  std::string parquet_path_;
  std::vector<uint8_t> expected_;
};

}  // namespace
#endif

TEST(TalonConfigTest, DefaultsAreDisabled) {
  api::Properties properties;
  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
  EXPECT_FALSE(config.talon_enabled);
  EXPECT_TRUE(config.talon_coordinator.empty());
  EXPECT_EQ(config.talon_block_size, 256U * 1024U * 1024U);
}

TEST(TalonConfigTest, ParsesExplicitProperties) {
  api::Properties properties;
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, "remote"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_ENABLED, "true"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, "127.0.0.1:7000"), std::nullopt);
  ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_BLOCK_SIZE, "8388608"), std::nullopt);

  ArrowFileSystemConfig config;
  ASSERT_STATUS_OK(ArrowFileSystemConfig::create_file_system_config(properties, config));
  EXPECT_TRUE(config.talon_enabled);
  EXPECT_EQ(config.talon_coordinator, "127.0.0.1:7000");
  EXPECT_EQ(config.talon_block_size, 8388608U);
}

TEST(TalonConfigTest, EnabledRequiresRemoteCoordinatorAndBlockSize) {
  for (const auto& [storage_type, coordinator, block_size] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"local", "127.0.0.1:7000", "8388608"}, {"remote", "", "8388608"}, {"remote", "127.0.0.1:7000", "0"}}) {
    api::Properties properties;
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_STORAGE_TYPE, storage_type.c_str()), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_ENABLED, "true"), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_COORDINATOR, coordinator.c_str()), std::nullopt);
    ASSERT_EQ(api::SetValue(properties, PROPERTY_FS_TALON_BLOCK_SIZE, block_size.c_str()), std::nullopt);
    ArrowFileSystemConfig config;
    EXPECT_FALSE(ArrowFileSystemConfig::create_file_system_config(properties, config).ok());
  }
}

TEST(TalonConfigTest, CacheKeyIncludesTalonRouting) {
  ArrowFileSystemConfig base;
  base.storage_type = "remote";
  base.cloud_provider = kCloudProviderAWS;
  base.address = "127.0.0.1:9000";
  base.bucket_name = "test-bucket";

  auto enabled = base;
  enabled.talon_enabled = true;
  enabled.talon_coordinator = "127.0.0.1:7000";
  enabled.talon_block_size = 8388608;
  EXPECT_NE(base.GetCacheKey(), enabled.GetCacheKey());

  auto other_coordinator = enabled;
  other_coordinator.talon_coordinator = "127.0.0.1:7001";
  EXPECT_NE(enabled.GetCacheKey(), other_coordinator.GetCacheKey());

  auto other_block_size = enabled;
  other_block_size.talon_block_size = 16777216;
  EXPECT_NE(enabled.GetCacheKey(), other_block_size.GetCacheKey());
}

TEST(TalonTestEnvTest, AddsTalonPropertiesWhenEnabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, "8388608");

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_TRUE(IsTalonEnv());
  EXPECT_TRUE(api::GetValue<bool>(properties, PROPERTY_FS_TALON_ENABLED).ValueOrDie());
  EXPECT_EQ(api::GetValue<std::string>(properties, PROPERTY_FS_TALON_COORDINATOR).ValueOrDie(), "127.0.0.1:7000");
  EXPECT_EQ(api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_BLOCK_SIZE).ValueOrDie(), 8388608U);
}

TEST(TalonTestEnvTest, AcceptsOneAndUsesDefaultBlockSize) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "1");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, nullptr);

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_TRUE(IsTalonEnv());
  EXPECT_EQ(api::GetValue<uint32_t>(properties, PROPERTY_FS_TALON_BLOCK_SIZE).ValueOrDie(), 268435456U);
}

TEST(TalonTestEnvTest, DoesNotAddTalonPropertiesWhenDisabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "false");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");
  ScopedEnv block_size(ENV_VAR_TALON_BLOCK_SIZE, "8388608");

  api::Properties properties;
  ASSERT_STATUS_OK(InitTestProperties(properties));
  EXPECT_FALSE(IsTalonEnv());
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_ENABLED));
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_COORDINATOR));
  EXPECT_FALSE(properties.contains(PROPERTY_FS_TALON_BLOCK_SIZE));
}

TEST(TalonTestEnvTest, RejectsEnabledTalonOutsideRemoteStorage) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "local");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, "127.0.0.1:7000");

  api::Properties properties;
  const auto status = InitTestProperties(properties);
  EXPECT_TRUE(status.IsInvalid()) << status.ToString();
}

TEST(TalonTestEnvTest, RequiresCoordinatorWhenEnabled) {
  ScopedEnv storage_type(ENV_VAR_STORAGE_TYPE, "remote");
  ScopedEnv enabled(ENV_VAR_TALON_ENABLED, "true");
  ScopedEnv coordinator(ENV_VAR_TALON_COORDINATOR, nullptr);

  api::Properties properties;
  const auto status = InitTestProperties(properties);
  EXPECT_TRUE(status.IsInvalid()) << status.ToString();
}

TEST(TalonFileSystemTest, S3CompatibleProviderIdentityIsPartOfEquality) {
  ArrowFileSystemConfig aws;
  aws.storage_type = "remote";
  aws.cloud_provider = kCloudProviderAWS;
  aws.address = "127.0.0.1:9000";
  aws.bucket_name = "test-bucket";
  aws.access_key_id = "access-key";
  aws.access_key_value = "secret-key";
  aws.region = "us-east-1";
  aws.use_ssl = false;

  auto aliyun = aws;
  aliyun.cloud_provider = kCloudProviderAliyun;

  ASSERT_AND_ASSIGN(auto aws_fs, CreateArrowFileSystem(aws));
  ASSERT_AND_ASSIGN(auto aliyun_fs, CreateArrowFileSystem(aliyun));
  const auto aws_proxy = std::dynamic_pointer_cast<FileSystemProxy>(aws_fs);
  const auto aliyun_proxy = std::dynamic_pointer_cast<FileSystemProxy>(aliyun_fs);
  ASSERT_NE(aws_proxy, nullptr);
  ASSERT_NE(aliyun_proxy, nullptr);
  EXPECT_EQ(aws_proxy->base_path(), "test-bucket/");
  EXPECT_EQ(aliyun_proxy->base_path(), "test-bucket/");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(aws_proxy->base_fs()), nullptr);
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(aliyun_proxy->base_fs()), nullptr);
  EXPECT_FALSE(aws_fs->Equals(*aliyun_fs));
  EXPECT_FALSE(aliyun_fs->Equals(*aws_fs));
}

#ifdef WITH_TALON
class TalonProviderEqualityTest : public ::testing::TestWithParam<const char*> {};

TEST_P(TalonProviderEqualityTest, ComparesRealProviderAndTalonSymmetrically) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = GetParam();
  config.address = config.cloud_provider == kCloudProviderAzure ? "core.windows.net" : "127.0.0.1:9000";
  config.bucket_name = "test-bucket";
  config.access_key_id = "testaccount";
  config.access_key_value = "c2VjcmV0";
  config.region = "us-east-1";

  ASSERT_AND_ASSIGN(auto disabled, CreateArrowFileSystem(config));
  ASSERT_AND_ASSIGN(auto same_disabled, CreateArrowFileSystem(config));
  EXPECT_TRUE(disabled->Equals(*same_disabled));
  EXPECT_TRUE(same_disabled->Equals(*disabled));

  config.talon_enabled = true;
  config.talon_coordinator = "127.0.0.1:7000";
  ASSERT_AND_ASSIGN(auto enabled, CreateArrowFileSystem(config));
  ASSERT_EQ(disabled->type_name(), enabled->type_name());
  EXPECT_FALSE(enabled->Equals(*disabled));
  EXPECT_FALSE(disabled->Equals(*enabled));

  const auto disabled_proxy = std::dynamic_pointer_cast<FileSystemProxy>(disabled);
  const auto enabled_proxy = std::dynamic_pointer_cast<FileSystemProxy>(enabled);
  ASSERT_NE(disabled_proxy, nullptr);
  ASSERT_NE(enabled_proxy, nullptr);
  EXPECT_FALSE(enabled_proxy->base_fs()->Equals(*disabled_proxy->base_fs()));
  EXPECT_FALSE(disabled_proxy->base_fs()->Equals(*enabled_proxy->base_fs()));
  // A raw provider must also reject its bucket-rooted wrapper despite the matching type name.
  EXPECT_FALSE(disabled_proxy->base_fs()->Equals(*disabled));

  ASSERT_AND_ASSIGN(auto same_enabled, CreateArrowFileSystem(config));
  EXPECT_TRUE(enabled->Equals(*same_enabled));
  EXPECT_TRUE(same_enabled->Equals(*enabled));

  config.talon_enabled = false;
  config.access_key_id = "differentaccount";
  ASSERT_AND_ASSIGN(auto different_account, CreateArrowFileSystem(config));
  EXPECT_FALSE(disabled->Equals(*different_account));
  EXPECT_FALSE(different_account->Equals(*disabled));
}

INSTANTIATE_TEST_SUITE_P(CloudProviders,
                         TalonProviderEqualityTest,
                         ::testing::Values(kCloudProviderAWS, kCloudProviderAzure));

TEST(TalonBridgeErrorTest, PreservesClassificationAndDiagnostics) {
  const std::string message =
      "all replicas failed after refresh; last worker 10.0.0.8:9000: worker error: Timeout: backend deadline exceeded";
  for (const auto& [talon_code, storage_code] : {
           std::pair{talon::ffi::TalonErrorCode::Timeout, ExtendStatusCode::StorageTransientTimeout},
           std::pair{talon::ffi::TalonErrorCode::Unavailable, ExtendStatusCode::StorageTransientService},
           std::pair{talon::ffi::TalonErrorCode::RateLimited, ExtendStatusCode::StorageTransientThrottling},
           std::pair{talon::ffi::TalonErrorCode::VersionMismatch, ExtendStatusCode::AwsErrorPreConditionFailed},
           std::pair{talon::ffi::TalonErrorCode::NonRetryable, ExtendStatusCode::AwsErrorNonRetryable},
       }) {
    const auto status =
        talon::internal::ToArrowIoResult("read", static_cast<int32_t>(talon_code), 0, message.c_str()).status();
    const auto detail = ExtendStatusDetail::UnwrapStatus(status);
    ASSERT_NE(detail, nullptr);
    EXPECT_EQ(detail->code(), storage_code);
    EXPECT_EQ(detail->retryable(), talon_code == talon::ffi::TalonErrorCode::Timeout ||
                                       talon_code == talon::ffi::TalonErrorCode::Unavailable ||
                                       talon_code == talon::ffi::TalonErrorCode::RateLimited);
    EXPECT_NE(status.message().find(message), std::string::npos);
  }
}

TEST(TalonBridgeErrorTest, HandlesInvalidMissingAndUnknownErrors) {
  EXPECT_TRUE(talon::internal::ToArrowIoResult(
                  "read", static_cast<int32_t>(talon::ffi::TalonErrorCode::InvalidArgument), 0, "bad range")
                  .status()
                  .IsInvalid());
  const auto missing = talon::internal::ToArrowIoResult(
                           "stat", static_cast<int32_t>(talon::ffi::TalonErrorCode::NotFound), 0, "missing object")
                           .status();
  EXPECT_EQ(arrow::internal::ErrnoFromStatus(missing), ENOENT);
  const auto unknown = talon::internal::ToArrowIoResult("read", 99, 0, nullptr).status();
  EXPECT_TRUE(unknown.IsIOError());
  EXPECT_EQ(unknown.detail(), nullptr);
  EXPECT_NE(unknown.message().find("unknown Talon error"), std::string::npos);
  EXPECT_EQ(talon::internal::ToArrowIoResult("read", 0, 42, nullptr).ValueOrDie(), 42);
  EXPECT_TRUE(
      talon::internal::ToArrowIoResult("stat", 0, std::numeric_limits<uint64_t>::max(), nullptr).status().IsIOError());
}

TEST(TalonBridgeTest, MapsClientCreationErrorAndPreservesTalonMessage) {
  const auto result = talon::TalonClient::Make("127.0.0.1:7000", 0);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsIOError()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to create Talon client"), std::string::npos);
  EXPECT_NE(result.status().message().find("block_size must be non-zero"), std::string::npos);
}

TEST(TalonBridgeTest, MapsOpenErrorAndPreservesTalonMessage) {
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make("127.0.0.1:7000", 8U * 1024U * 1024U));

  const auto result = client->OpenObject("aws", "test-bucket", "", std::nullopt);

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsIOError()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to open Talon object"), std::string::npos);
  EXPECT_NE(result.status().message().find("bucket and object key must be non-empty"), std::string::npos);
}

TEST(TalonBridgeTest, MapsAsyncErrorAndPreservesTalonMessage) {
  ASSERT_AND_ASSIGN(auto client, talon::TalonClient::Make("127.0.0.1:7000", 8U * 1024U * 1024U));
  ASSERT_AND_ASSIGN(auto reader, client->OpenObject("aws", "test-bucket", "path/a", talon::TalonObjectStat{0, ""}));

  const auto result = reader.ReadAtAsync(0, std::numeric_limits<uint64_t>::max(), nullptr).result();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid()) << result.status().ToString();
  EXPECT_NE(result.status().message().find("Failed to read Talon object"), std::string::npos);
  EXPECT_NE(result.status().message().find("Talon read length exceeds isize::MAX"), std::string::npos);
}

TEST(TalonFileSystemTest, PreservesOuterSubtreeAndKnownSize) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.cloud_provider = kCloudProviderAWS;
  config.address = "http://127.0.0.1:9000";
  config.bucket_name = "test-bucket";
  config.access_key_id = "minioadmin";
  config.access_key_value = "minioadmin";
  config.region = "us-east-1";
  config.s3_crt_async_read = false;
  config.talon_enabled = true;
  config.talon_coordinator = "127.0.0.1:7000";
  config.talon_block_size = 8388608;

  ASSERT_AND_ASSIGN(auto fs, CreateArrowFileSystem(config));
  auto proxy = std::dynamic_pointer_cast<FileSystemProxy>(fs);
  ASSERT_NE(proxy, nullptr);
  EXPECT_EQ(proxy->base_path(), "test-bucket/");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(proxy->base_fs()), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<UploadConditional>(fs), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<UploadSizable>(fs), nullptr);
  EXPECT_NE(std::dynamic_pointer_cast<Observable>(fs), nullptr);

  arrow::fs::FileInfo info("known-empty", arrow::fs::FileType::File);
  info.set_size(0);
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(auto size, file->GetSize());
  EXPECT_EQ(size, 0);
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);
  ASSERT_AND_ASSIGN(const auto async_size, async_file->GetSizeAsync().result());
  EXPECT_EQ(async_size, 0);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ProducerAcceptsRawProviderFilesystem) {
  const auto config = Config();
  auto origin = std::make_shared<RecordingFileSystem>("origin");

  ASSERT_AND_ASSIGN(auto talon_fs, TalonFileSystemProducer(config, origin).Make());
  EXPECT_EQ(talon_fs->type_name(), "recording");
  EXPECT_EQ(std::dynamic_pointer_cast<FileSystemProxy>(talon_fs), nullptr);
}

TEST_F(TalonFileSystemServiceFreeTest, OpenWithFileInfoRejectsNonFiles) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (const auto type : {arrow::fs::FileType::NotFound, arrow::fs::FileType::Directory}) {
    SCOPED_TRACE(static_cast<int>(type));
    const arrow::fs::FileInfo info("prefix/object", type);
    const auto file = fs->OpenInputFile(info);
    const auto stream = fs->OpenInputStream(info);
    for (const auto& status : {file.status(), stream.status()}) {
      EXPECT_TRUE(status.IsIOError()) << status;
      EXPECT_NE(status.message().find("test-bucket/prefix/object"), std::string::npos);
      if (type == arrow::fs::FileType::NotFound) {
        EXPECT_EQ(arrow::internal::ErrnoFromStatus(status), ENOENT);
      } else {
        EXPECT_NE(status.message().find("Not a regular file"), std::string::npos);
      }
    }
  }
  EXPECT_EQ(origin->input_open_calls, 0);
}

TEST_F(TalonFileSystemServiceFreeTest, OpenWithFileInfoAcceptsFileAndUnknown) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  for (const auto type : {arrow::fs::FileType::File, arrow::fs::FileType::Unknown}) {
    for (const int64_t size : {-1, 0, 7}) {
      SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << ", size=" << size);
      arrow::fs::FileInfo info("prefix/object", type);
      info.set_size(size);
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
      ASSERT_AND_ASSIGN(auto stream, fs->OpenInputStream(info));
      if (size >= 0) {
        ASSERT_AND_ASSIGN(const auto known_size, file->GetSize());
        EXPECT_EQ(known_size, size);
      }
      ASSERT_STATUS_OK(file->Close());
      ASSERT_STATUS_OK(stream->Close());
    }
  }
  EXPECT_EQ(origin->input_open_calls, 0);
}

TEST_F(TalonFileSystemServiceFreeTest, EquivalentTalonFilesystemsCompareEqualSymmetrically) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto left, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  ASSERT_AND_ASSIGN(auto right, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  const auto left_proxy = std::dynamic_pointer_cast<FileSystemProxy>(left);
  const auto right_proxy = std::dynamic_pointer_cast<FileSystemProxy>(right);
  ASSERT_NE(left_proxy, nullptr);
  ASSERT_NE(right_proxy, nullptr);
  const auto left_talon = left_proxy->base_fs();
  const auto right_talon = right_proxy->base_fs();

  EXPECT_TRUE(left_talon->Equals(*right_talon));
  EXPECT_TRUE(right_talon->Equals(*left_talon));
}

TEST_F(TalonFileSystemServiceFreeTest, PreservesProviderTypeName) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));

  EXPECT_EQ(fs->type_name(), "recording");
}

TEST_F(TalonFileSystemServiceFreeTest, TalonFilesystemComparesUnequalToUndecoratedOrigin) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto talon_fs, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  auto raw_origin = std::make_shared<RecordingFileSystem>("origin");

  const auto talon_proxy = std::dynamic_pointer_cast<FileSystemProxy>(talon_fs);
  ASSERT_NE(talon_proxy, nullptr);
  const auto talon_base = talon_proxy->base_fs();
  EXPECT_FALSE(talon_base->Equals(*raw_origin));
  EXPECT_FALSE(raw_origin->Equals(*talon_base));
}

TEST_F(TalonFileSystemServiceFreeTest, EqualityIncludesTalonRouting) {
  const auto config = Config();
  ASSERT_AND_ASSIGN(auto base, Wrap(config, std::make_shared<RecordingFileSystem>("origin")));
  const auto base_proxy = std::dynamic_pointer_cast<FileSystemProxy>(base);
  ASSERT_NE(base_proxy, nullptr);
  const auto base_talon = base_proxy->base_fs();

  auto other_coordinator = config;
  other_coordinator.talon_coordinator = "127.0.0.1:7001";
  ASSERT_AND_ASSIGN(auto coordinator_fs, Wrap(other_coordinator, std::make_shared<RecordingFileSystem>("origin")));
  const auto coordinator_proxy = std::dynamic_pointer_cast<FileSystemProxy>(coordinator_fs);
  ASSERT_NE(coordinator_proxy, nullptr);
  const auto coordinator_talon = coordinator_proxy->base_fs();
  EXPECT_FALSE(base_talon->Equals(*coordinator_talon));
  EXPECT_FALSE(coordinator_talon->Equals(*base_talon));

  auto other_block_size = config;
  other_block_size.talon_block_size *= 2;
  ASSERT_AND_ASSIGN(auto block_size_fs, Wrap(other_block_size, std::make_shared<RecordingFileSystem>("origin")));
  const auto block_size_proxy = std::dynamic_pointer_cast<FileSystemProxy>(block_size_fs);
  ASSERT_NE(block_size_proxy, nullptr);
  const auto block_size_talon = block_size_proxy->base_fs();
  EXPECT_FALSE(base_talon->Equals(*block_size_talon));
  EXPECT_FALSE(block_size_talon->Equals(*base_talon));
}

TEST_F(TalonFileSystemServiceFreeTest, ForwardsStreamingFileInfoGeneratorWithFullSelector) {
  const auto config = Config();
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(config, origin));

  arrow::fs::FileSelector selector;
  selector.base_dir = "prefix";
  selector.allow_not_found = true;
  selector.recursive = true;
  selector.max_recursion = 3;
  auto generator = fs->GetFileInfoGenerator(selector);

  EXPECT_EQ(origin->synchronous_listing_calls(), 0);
  EXPECT_EQ(origin->generator_calls(), 1);
  EXPECT_EQ(origin->last_generator_selector().base_dir, "test-bucket/prefix");
  EXPECT_TRUE(origin->last_generator_selector().allow_not_found);
  EXPECT_TRUE(origin->last_generator_selector().recursive);
  EXPECT_EQ(origin->last_generator_selector().max_recursion, 3);

  ASSERT_AND_ASSIGN(auto first, generator().result());
  ASSERT_EQ(first.size(), 1);
  EXPECT_EQ(first[0].path(), "prefix/first");

  ASSERT_AND_ASSIGN(auto second, generator().result());
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(second[0].path(), "prefix/second");

  ASSERT_AND_ASSIGN(auto end, generator().result());
  EXPECT_TRUE(end.empty());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataLazilyPreservesOriginMetadata) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
  info.set_size(7);
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, 7);
  EXPECT_EQ(origin->input_open_calls, 0);

  ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
  ASSERT_NE(metadata, nullptr);
  ASSERT_AND_ASSIGN(const auto etag, metadata->Get("ETag"));
  ASSERT_AND_ASSIGN(const auto content_length, metadata->Get("Content-Length"));
  ASSERT_AND_ASSIGN(const auto owner, metadata->Get("owner"));
  EXPECT_EQ(etag, "\"origin-etag\"");
  EXPECT_EQ(content_length, "7");
  EXPECT_EQ(owner, "test-owner");
  EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
  ASSERT_AND_ASSIGN(auto repeated, file->ReadMetadata());
  EXPECT_EQ(repeated, metadata);
  ASSERT_AND_ASSIGN(auto async_metadata, file->ReadMetadataAsync().result());
  EXPECT_EQ(async_metadata, metadata);
  EXPECT_EQ(origin->input_open_calls, 1);

  ASSERT_STATUS_OK(file->Close());
  EXPECT_TRUE(origin->input_file->closed());
  EXPECT_TRUE(file->ReadMetadata().status().IsInvalid());
  EXPECT_TRUE(file->ReadMetadataAsync().result().status().IsInvalid());
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataAsyncForwardsProviderFutureAndContext) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputStream("prefix/object"));
  arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
  origin->input_file->async_result = arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>>::Make();

  auto future = file->ReadMetadataAsync(arrow::io::IOContext(&pool));
  EXPECT_FALSE(future.is_finished());
  EXPECT_EQ(origin->input_file->async_pool, &pool);
  EXPECT_EQ(origin->last_input_path, "test-bucket/prefix/object");
  origin->input_file->async_result.MarkFinished(origin->input_file->metadata_result);
  ASSERT_AND_ASSIGN(auto metadata, future.result());
  ASSERT_NE(metadata, nullptr);
  ASSERT_AND_ASSIGN(const auto etag, metadata->Get("ETag"));
  EXPECT_EQ(etag, "\"origin-etag\"");
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, ReadMetadataPreservesOriginErrorsAndNullMetadata) {
  auto origin = std::make_shared<RecordingFileSystem>("origin");
  ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
  ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));
  const auto error =
      MakeExtendError(ExtendStatusCode::AwsErrorAccessDenied, "origin metadata denied", "provider detail");
  origin->input_open_status = error;
  EXPECT_EQ(file->ReadMetadata().status(), error);
  EXPECT_EQ(file->ReadMetadataAsync().result().status(), error);

  origin->input_open_status = arrow::Status::OK();
  origin->input_file->metadata_result = error;
  EXPECT_EQ(file->ReadMetadata().status(), error);
  EXPECT_EQ(file->ReadMetadataAsync().result().status(), error);

  origin->input_file->metadata_result = std::shared_ptr<const arrow::KeyValueMetadata>{};
  ASSERT_AND_ASSIGN(auto metadata, file->ReadMetadata());
  EXPECT_EQ(metadata, nullptr);
  ASSERT_AND_ASSIGN(auto async_metadata, file->ReadMetadataAsync().result());
  EXPECT_EQ(async_metadata, nullptr);
  ASSERT_STATUS_OK(file->Close());
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsClampKnownSizeBeforeAllocation) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
    info.set_size(8192);
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

    const auto result =
        async ? file->ReadAsync(io_context, 4096, 64 * 1024 * 1024).result() : file->ReadAt(4096, 64 * 1024 * 1024);
    EXPECT_TRUE(result.status().IsOutOfMemory()) << result.status();
    EXPECT_EQ(pool.last_requested_size, 4096);
    ASSERT_STATUS_OK(file->Close());
  }
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsAtKnownEofDoNotAllocatePayload) {
  for (const bool async : {false, true}) {
    for (const int64_t size : {0, 8192}) {
      SCOPED_TRACE(::testing::Message() << "async=" << async << ", size=" << size);
      arrow::ProxyMemoryPool pool(arrow::default_memory_pool());
      const arrow::io::IOContext io_context(&pool);
      auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
      ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
      arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
      info.set_size(size);
      ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

      ASSERT_AND_ASSIGN(auto buffer, async ? file->ReadAsync(io_context, size, 64 * 1024 * 1024).result()
                                           : file->ReadAt(size, 64 * 1024 * 1024));
      EXPECT_EQ(buffer->size(), 0);
      EXPECT_EQ(pool.max_memory(), 0);
      ASSERT_STATUS_OK(file->Close());
    }
  }
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsValidateBeforeAllocation) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    arrow::fs::FileInfo info("prefix/object", arrow::fs::FileType::File);
    info.set_size(8192);
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile(info));

    const std::vector<std::tuple<int64_t, int64_t, arrow::StatusCode>> cases{
        {-1, 64 * 1024 * 1024, arrow::StatusCode::Invalid},
        {0, -1, arrow::StatusCode::Invalid},
        {8193, 64 * 1024 * 1024, arrow::StatusCode::IOError}};
    for (const auto& [position, nbytes, expected_code] : cases) {
      const auto result =
          async ? file->ReadAsync(io_context, position, nbytes).result() : file->ReadAt(position, nbytes);
      EXPECT_EQ(result.status().code(), expected_code);
    }
    ASSERT_STATUS_OK(file->Close());
    const auto closed =
        async ? file->ReadAsync(io_context, 0, 64 * 1024 * 1024).result() : file->ReadAt(0, 64 * 1024 * 1024);
    EXPECT_TRUE(closed.status().IsInvalid());
    EXPECT_EQ(pool.last_requested_size, -1);
  }
}

TEST_F(TalonFileSystemServiceFreeTest, AllocatingReadsWithUnknownSizeDoNotPreflightStat) {
  for (const bool async : {false, true}) {
    SCOPED_TRACE(async);
    RejectingMemoryPool pool;
    const arrow::io::IOContext io_context(&pool);
    auto origin = std::make_shared<RecordingFileSystem>("origin", io_context);
    ASSERT_AND_ASSIGN(auto fs, Wrap(Config(), origin));
    ASSERT_AND_ASSIGN(auto file, fs->OpenInputFile("prefix/object"));

    const auto result =
        async ? file->ReadAsync(io_context, 4096, 64 * 1024 * 1024).result() : file->ReadAt(4096, 64 * 1024 * 1024);
    EXPECT_TRUE(result.status().IsOutOfMemory()) << result.status();
    EXPECT_EQ(pool.last_requested_size, 64 * 1024 * 1024);
    ASSERT_STATUS_OK(file->Close());
  }
}

TEST_F(TalonIntegrationTest, WriteOriginReadThroughTalon) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(buffer->size(), static_cast<int64_t>(expected_.size()));
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data(), expected_.size()), 0);
}

TEST_F(TalonIntegrationTest, OpenWithKnownSize) {
  ASSERT_AND_ASSIGN(const auto info, fs_->GetFileInfo(path_));
  ASSERT_EQ(info.type(), arrow::fs::FileType::File);
  ASSERT_EQ(info.size(), static_cast<int64_t>(expected_.size()));
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(info));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  EXPECT_EQ(size, static_cast<int64_t>(expected_.size()));
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(0, static_cast<int64_t>(expected_.size())));
  ASSERT_EQ(buffer->size(), static_cast<int64_t>(expected_.size()));
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data(), expected_.size()), 0);
}

TEST_F(TalonIntegrationTest, OpenWithoutKnownSize) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  ASSERT_AND_ASSIGN(const auto size, file->GetSize());
  ASSERT_EQ(size, static_cast<int64_t>(expected_.size()));

  constexpr int64_t kOffset = 4096;
  constexpr int64_t kLength = 4096;
  ASSERT_AND_ASSIGN(auto buffer, file->ReadAt(kOffset, kLength));
  ASSERT_EQ(buffer->size(), kLength);
  EXPECT_EQ(std::memcmp(buffer->data(), expected_.data() + kOffset, kLength), 0);
}

TEST_F(TalonIntegrationTest, AsyncReadAtIntoCallerBuffer) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  constexpr int64_t kOffset = 1024;
  std::vector<uint8_t> output(4096);
  auto future = async_file->ReadAtAsyncInto(kOffset, static_cast<int64_t>(output.size()), output.data());
  ASSERT_AND_ASSIGN(const auto bytes_read, future.result());
  ASSERT_EQ(bytes_read, static_cast<int64_t>(output.size()));
  EXPECT_TRUE(std::equal(output.begin(), output.end(), expected_.begin() + kOffset));
}

TEST_F(TalonIntegrationTest, ConcurrentAsyncRanges) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  auto* const async_file = dynamic_cast<NonBlockingRandomAccessFile*>(file.get());
  ASSERT_NE(async_file, nullptr);

  constexpr size_t kRangeCount = 16;
  constexpr size_t kRangeSize = 64 * 1024;
  std::vector<std::vector<uint8_t>> outputs(kRangeCount, std::vector<uint8_t>(kRangeSize));
  std::vector<arrow::Future<int64_t>> futures;
  futures.reserve(kRangeCount);
  for (size_t i = 0; i < kRangeCount; ++i) {
    futures.push_back(async_file->ReadAtAsyncInto(static_cast<int64_t>(i * kRangeSize),
                                                  static_cast<int64_t>(kRangeSize), outputs[i].data()));
  }

  // Drain all reads before a fatal assertion can release buffers still used by Rust.
  // Each future retains its result for the assertions below, including failures.
  for (const auto& future : futures) {
    future.Wait();
  }

  for (size_t i = 0; i < kRangeCount; ++i) {
    ASSERT_AND_ASSIGN(const auto bytes_read, futures[i].result());
    ASSERT_EQ(bytes_read, static_cast<int64_t>(kRangeSize));
    EXPECT_TRUE(std::equal(outputs[i].begin(), outputs[i].end(), expected_.begin() + i * kRangeSize));
  }
}

TEST_F(TalonIntegrationTest, EofAndClosedFileSemantics) {
  ASSERT_AND_ASSIGN(auto file, fs_->OpenInputFile(path_));
  std::vector<uint8_t> output(4096);
  ASSERT_AND_ASSIGN(const auto bytes_read, file->ReadAt(static_cast<int64_t>(expected_.size()),
                                                        static_cast<int64_t>(output.size()), output.data()));
  EXPECT_EQ(bytes_read, 0);

  ASSERT_STATUS_OK(file->Close());
  const auto closed_read = file->ReadAt(0, static_cast<int64_t>(output.size()), output.data());
  ASSERT_FALSE(closed_read.ok());
  EXPECT_TRUE(closed_read.status().IsInvalid()) << closed_read.status().ToString();
}

TEST_F(TalonIntegrationTest, ParquetReadThroughNormalStorageApi) {
  ASSERT_STATUS_OK(CreateTestDir(fs_, parquet_path_));
  ASSERT_AND_ASSIGN(auto schema, CreateTestSchema());
  ASSERT_AND_ASSIGN(auto batch, CreateTestData(schema));
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(LOON_FORMAT_PARQUET, schema));

  auto writer = api::Writer::create(parquet_path_, schema, std::move(policy), properties_);
  ASSERT_NE(writer, nullptr);
  ASSERT_STATUS_OK(writer->write(batch));
  ASSERT_AND_ASSIGN(auto column_groups, writer->close());

  auto reader = api::Reader::create(column_groups, schema, nullptr, properties_);
  ASSERT_NE(reader, nullptr);
  ASSERT_AND_ASSIGN(auto batch_reader, reader->get_record_batch_reader());
  std::shared_ptr<arrow::RecordBatch> result;
  ASSERT_STATUS_OK(batch_reader->ReadNext(&result));
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->num_rows(), batch->num_rows());
  ASSERT_STATUS_OK(ValidateRowAlignment(result));
  ASSERT_STATUS_OK(batch_reader->ReadNext(&result));
  EXPECT_EQ(result, nullptr);
}
#endif

#ifndef WITH_TALON
TEST(TalonFileSystemTest, EnabledWithoutBuildSupportIsRejected) {
  ArrowFileSystemConfig config;
  config.storage_type = "remote";
  config.talon_enabled = true;
  config.talon_coordinator = "127.0.0.1:7000";
  auto result = CreateArrowFileSystem(config);
  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().IsInvalid());
}
#endif

}  // namespace milvus_storage::test
