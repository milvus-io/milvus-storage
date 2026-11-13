#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <unistd.h>

#include "milvus-storage/filesystem/fs.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/filesystem/s3/s3_global.h"

namespace milvus_storage {
namespace {

// Exit code, initialize S3 before using the cache, explicitly finalize S3.
class S3ShutdownDeathTest : public ::testing::TestWithParam<std::tuple<int, bool, bool>> {};

TEST_P(S3ShutdownDeathTest, ReleasesCachedCredentialsBeforeSdkShutdown) {
  const auto [exit_code, s3_first, explicit_finalize] = GetParam();
  const auto original_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        // Bound both SDK initialization and exit handlers; a regression must fail
        // the test instead of hanging the test runner indefinitely.
        ::alarm(20);
        auto run = [&] {
          ASSERT_EQ(::setenv("AWS_EC2_METADATA_DISABLED", "true", 1), 0);
          ASSERT_EQ(::setenv("AWS_CONFIG_FILE", "/dev/null", 1), 0);
          ASSERT_EQ(::setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null", 1), 0);
          ASSERT_EQ(::setenv("AWS_REGION", "us-west-2", 1), 0);
          ASSERT_EQ(::setenv("AWS_DEFAULT_REGION", "us-west-2", 1), 0);
          ASSERT_EQ(::setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/test-irsa", 1), 0);
          ASSERT_EQ(::setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/dev/null", 1), 0);

          // Construct real WebIdentity providers without fetching credentials or
          // making any S3 requests. Their HTTP pools retain the CRT bootstrap.
          api::Properties properties;
          properties[PROPERTY_FS_STORAGE_TYPE] = std::string("remote");
          properties[PROPERTY_FS_CLOUD_PROVIDER] = std::string("aws");
          properties[PROPERTY_FS_ADDRESS] = std::string("127.0.0.1:1");
          properties[PROPERTY_FS_BUCKET_NAME] = std::string("test-bucket");
          properties[PROPERTY_FS_REGION] = std::string("us-west-2");
          properties[PROPERTY_FS_USE_SSL] = false;
          properties[PROPERTY_FS_ROLE_ARN] = std::string("arn:aws:iam::123456789012:role/test-target");
          const auto config = std::make_shared<const api::Properties>(std::move(properties));

          if (s3_first) {
            S3FileSystemProducer producer(ArrowFileSystemConfig{});
            const auto status = producer.InitS3();
            ASSERT_TRUE(status.ok()) << status.ToString();
          }
          auto& cache = FilesystemCache::getInstance();
          std::weak_ptr<arrow::fs::FileSystem> filesystem;
          try {
            auto result = cache.get(config);
            ASSERT_TRUE(result.ok()) << result.status().ToString();
            auto fs = std::move(result).ValueOrDie();
            filesystem = fs;
            if (exit_code != 0) {
              throw std::runtime_error("simulated application failure");
            }
          } catch (const std::runtime_error&) {
            ASSERT_NE(exit_code, 0);
          }
          // Both normal completion and exception unwinding release the caller's
          // handle, leaving the cache as the only owner.
          ASSERT_EQ(filesystem.use_count(), 1);
          ASSERT_EQ(cache.size(), 1);
          if (explicit_finalize) {
            auto status = FinalizeS3();
            ASSERT_TRUE(status.ok()) << status.ToString();
            ASSERT_TRUE(filesystem.expired());
            ASSERT_EQ(cache.size(), 0);
            status = FinalizeS3();
            ASSERT_TRUE(status.ok()) << status.ToString();
          }
        };
        run();
        if (::testing::Test::HasFailure()) {
          std::_Exit(100);
        }
        // Unlike _Exit, exit executes the native atexit handlers and static
        // destructors. Preserve the application's success or failure exit code.
        std::exit(exit_code);
      },
      ::testing::ExitedWithCode(exit_code), "");
  GTEST_FLAG_SET(death_test_style, original_style);
}

INSTANTIATE_TEST_SUITE_P(ExitPaths,
                         S3ShutdownDeathTest,
                         ::testing::Combine(::testing::Values(0, 42), ::testing::Bool(), ::testing::Bool()));

}  // namespace
}  // namespace milvus_storage
