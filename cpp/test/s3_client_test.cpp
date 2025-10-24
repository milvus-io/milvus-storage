/* c++
  File: cpp/include/milvus-storage/filesystem/s3/test_s3_client.h
  Minimal GoogleTest unit tests for milvus_storage::S3Client connecting to MinIO/S3.
*/
#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <memory>
#include <string>
#include <mutex>
#include <unistd.h>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>

#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_fs.h"

#include "test_util.h"

namespace milvus_storage {
namespace test {

class S3ClientTest : public ::testing::Test {
  protected:
  void SetUp() override {
    storage_type_ = GetEnvVar(ENV_VAR_STORAGE_TYPE).ValueOr("");
    access_key_id_ = GetEnvVar(ENV_VAR_ACCESS_KEY_ID).ValueOr("");
    access_key_value_ = GetEnvVar(ENV_VAR_ACCESS_KEY_VALUE).ValueOr("");
    address_ = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
    bucket_ = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("");
    region_ = GetEnvVar(ENV_VAR_REGION).ValueOr("");

    if (storage_type_.empty() || access_key_id_.empty() || access_key_value_.empty() || address_.empty() ||
        bucket_.empty() || region_.empty()) {
      GTEST_SKIP() << "S3 credentials not set. Please set environment variables:\n"
                   << "STORAGE_TYPE, ACCESS_KEY, SECRET_KEY, ADDRESS, BUCKET_NAME, REGION";
    }

    // Build ArrowFileSystemConfig and use S3FileSystemProducer to create
    // an ExtendedS3Options suitable for ClientBuilder.
    milvus_storage::ArrowFileSystemConfig fs_config;
    // Parse address to strip scheme if present
    fs_config.storage_type = storage_type_;
    fs_config.address = address_;
    fs_config.bucket_name = bucket_;
    fs_config.access_key_id = access_key_id_;
    fs_config.access_key_value = access_key_value_;
    fs_config.region = region_;

    milvus_storage::S3FileSystemProducer producer(fs_config);
    producer.InitS3();
    ASSERT_AND_ASSIGN(auto s3_options, producer.CreateS3Options());

    // Build an S3Client via ClientBuilder
    milvus_storage::ClientBuilder builder(s3_options);
    ASSERT_AND_ASSIGN(client_holder_, builder.BuildClient());
  }

  std::string storage_type_;
  std::string address_;
  std::string bucket_;
  std::string access_key_id_;
  std::string access_key_value_;
  std::string region_;
  Aws::SDKOptions sdk_options_;
  std::shared_ptr<S3ClientHolder> client_holder_;
};

// Test simple PutObject + GetObject roundtrip against MinIO/S3.
TEST_F(S3ClientTest, PutGetObjectRoundTrip) {
  const std::string key = "unittest/test_put_get.txt";
  const std::string content = "hello milvus-storage s3 client";

  {
    ASSERT_AND_ASSIGN(auto client_lock, client_holder_->Lock());
    Aws::S3::Model::PutObjectRequest put_request;
    put_request.SetBucket(bucket_.c_str());
    put_request.SetKey(key.c_str());

    auto ss = Aws::MakeShared<Aws::StringStream>("S3Test");
    (*ss) << content;
    put_request.SetBody(ss);

    auto put_outcome = client_lock.Move()->PutObject(put_request);
    ASSERT_TRUE(put_outcome.IsSuccess()) << "PutObject failed: " << put_outcome.GetError().GetMessage();
  }

  {
    ASSERT_AND_ASSIGN(auto client_lock, client_holder_->Lock());

    Aws::S3::Model::GetObjectRequest get_request;
    get_request.SetBucket(bucket_.c_str());
    get_request.SetKey(key.c_str());

    auto get_outcome = client_lock.Move()->GetObject(get_request);
    ASSERT_TRUE(get_outcome.IsSuccess()) << "GetObject failed: " << get_outcome.GetError().GetMessage();

    auto& stream = get_outcome.GetResult().GetBody();
    std::ostringstream oss;
    oss << stream.rdbuf();
    std::string downloaded = oss.str();

    EXPECT_EQ(downloaded, content);
  }
}

// Test CreateMultipartUpload followed by AbortMultipartUpload to ensure multipart flows can be created/cleaned.
TEST_F(S3ClientTest, CreateAndAbortMultipartUpload) {
  const std::string key = "unittest/test_multipart.txt";

  Aws::S3::Model::CreateMultipartUploadRequest create_request;
  create_request.SetBucket(bucket_.c_str());
  create_request.SetKey(key.c_str());

  ASSERT_AND_ASSIGN(auto client_lock, client_holder_->Lock());

  auto create_outcome = client_lock.Move()->CreateMultipartUpload(create_request);
  ASSERT_TRUE(create_outcome.IsSuccess()) << "CreateMultipartUpload failed: " << create_outcome.GetError().GetMessage();

  auto upload_id = create_outcome.GetResult().GetUploadId();
  ASSERT_FALSE(upload_id.empty());

  Aws::S3::Model::AbortMultipartUploadRequest abort_request;
  abort_request.SetBucket(bucket_.c_str());
  abort_request.SetKey(key.c_str());
  abort_request.SetUploadId(upload_id);

  ASSERT_AND_ASSIGN(auto client_lock2, client_holder_->Lock());

  auto abort_outcome = client_lock2.Move()->AbortMultipartUpload(abort_request);
  ASSERT_TRUE(abort_outcome.IsSuccess()) << "AbortMultipartUpload failed: " << abort_outcome.GetError().GetMessage();
}

TEST_F(S3ClientTest, TestConcurrent) {
  const int num_threads = 10;
  std::vector<std::thread> threads;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this]() {
      ASSERT_AND_ASSIGN(auto client_lock, client_holder_->Lock());
      sleep(1);
      client_lock.Move();  // do nothing
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  EXPECT_GT(duration.count(), 1.0 * 1000000);  // should be more than 1 second
  EXPECT_LT(duration.count(), 2.0 * 1000000);  // should be less than 2 seconds
}

}  // namespace test
}  // namespace milvus_storage
