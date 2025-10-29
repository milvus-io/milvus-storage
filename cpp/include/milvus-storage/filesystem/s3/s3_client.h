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

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include <arrow/result.h>
#include <arrow/io/interfaces.h>
#include <arrow/filesystem/s3fs.h>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>

namespace milvus_storage {

class S3ClientMetrics {
  public:
  S3ClientMetrics() = default;
  ~S3ClientMetrics() = default;

  inline void IncrementMultiPartUploadCreated() { multi_part_upload_created.fetch_add(1, std::memory_order_relaxed); }
  inline void IncrementMultiPartUploadFinished() { multi_part_upload_finished.fetch_add(1, std::memory_order_relaxed); }
  inline void IncrementUploadBytes(int64_t bytes) { upload_bytes.fetch_add(bytes, std::memory_order_relaxed); }
  inline void IncrementDownloadBytes(int64_t bytes) { download_bytes.fetch_add(bytes, std::memory_order_relaxed); }
  inline void IncrementUploadCount() { upload_count.fetch_add(1, std::memory_order_relaxed); }
  inline void IncrementDownloadCount() { download_count.fetch_add(1, std::memory_order_relaxed); }
  inline void IncrementFailedCount() { failed_count.fetch_add(1, std::memory_order_relaxed); }

  inline int64_t GetMultiPartUploadCreated() const { return multi_part_upload_created.load(std::memory_order_relaxed); }
  inline int64_t GetMultiPartUploadFinished() const {
    return multi_part_upload_finished.load(std::memory_order_relaxed);
  }
  inline int64_t GetUploadCount() const { return upload_count.load(std::memory_order_relaxed); }
  inline int64_t GetDownloadCount() const { return download_count.load(std::memory_order_relaxed); }
  inline int64_t GetUploadBytes() const { return upload_bytes.load(std::memory_order_relaxed); }
  inline int64_t GetDownloadBytes() const { return download_bytes.load(std::memory_order_relaxed); }
  inline int64_t GetFailedCount() const { return failed_count.load(std::memory_order_relaxed); }

  void Reset() {
    multi_part_upload_created.store(0, std::memory_order_relaxed);
    multi_part_upload_finished.store(0, std::memory_order_relaxed);
    upload_count.store(0, std::memory_order_relaxed);
    download_count.store(0, std::memory_order_relaxed);
    upload_bytes.store(0, std::memory_order_relaxed);
    download_bytes.store(0, std::memory_order_relaxed);
    failed_count.store(0, std::memory_order_relaxed);
  }

  private:
  std::atomic<int64_t> multi_part_upload_created{0};
  std::atomic<int64_t> multi_part_upload_finished{0};
  std::atomic<int64_t> upload_count{0};
  std::atomic<int64_t> download_count{0};
  std::atomic<int64_t> failed_count{0};

  std::atomic<int64_t> upload_bytes{0};
  std::atomic<int64_t> download_bytes{0};
};

class S3Client : public Aws::S3::S3Client {
  public:
  using Aws::S3::S3Client::S3Client;
  std::string GetBucketRegionFromHeaders(const Aws::Http::HeaderValueCollection& headers);

  template <typename ErrorType>
  arrow::Result<std::string> GetBucketRegionFromError(const std::string& bucket,
                                                      const Aws::Client::AWSError<ErrorType>& error);

  arrow::Result<std::string> GetBucketRegion(const std::string& bucket,
                                             const Aws::S3::Model::HeadBucketRequest& request);

  arrow::Result<std::string> GetBucketRegion(const std::string& bucket);

  Aws::S3::Model::CompleteMultipartUploadOutcome CompleteMultipartUploadWithErrorFixup(
      Aws::S3::Model::CompleteMultipartUploadRequest&& request) const;

  // Metrics related functions
  Aws::S3::Model::CreateMultipartUploadOutcome CreateMultipartUpload(
      const Aws::S3::Model::CreateMultipartUploadRequest& request) const override;

  Aws::S3::Model::UploadPartOutcome UploadPart(const Aws::S3::Model::UploadPartRequest& request) const override;
  Aws::S3::Model::PutObjectOutcome PutObject(const Aws::S3::Model::PutObjectRequest& request) const override;
  Aws::S3::Model::GetObjectOutcome GetObject(const Aws::S3::Model::GetObjectRequest& request) const override;

  std::shared_ptr<S3ClientMetrics> GetMetrics() const;

  public:
  std::shared_ptr<arrow::fs::S3RetryStrategy> s3_retry_strategy_;

  private:
  std::shared_ptr<S3ClientMetrics> metrics_ = std::make_shared<S3ClientMetrics>();
};

class S3ClientLock {
  public:
  S3Client* get();
  S3Client* operator->();

  // Move this S3ClientLock into a temporary instance
  S3ClientLock Move();

  protected:
  friend class S3ClientHolder;

  // Locks the finalizer until the S3ClientLock gets out of scope.
  std::shared_lock<std::shared_mutex> lock_;
  std::shared_ptr<S3Client> client_;
};

class S3ClientFinalizer;

class S3ClientHolder {
  public:
  /// \brief Return a RAII guard guaranteeing a S3Client is safe for use
  ///
  /// S3 finalization will be deferred until the returned S3ClientLock
  /// goes out of scope.
  /// An error is returned if S3 is already finalized.
  arrow::Result<S3ClientLock> Lock();

  S3ClientHolder(std::weak_ptr<S3ClientFinalizer> finalizer, std::shared_ptr<S3Client> client);

  void Finalize();

  protected:
  std::mutex mutex_;
  std::weak_ptr<S3ClientFinalizer> finalizer_;
  std::shared_ptr<S3Client> client_;
};

class S3ClientFinalizer : public std::enable_shared_from_this<S3ClientFinalizer> {
  using ClientHolderList = std::vector<std::weak_ptr<S3ClientHolder>>;

  public:
  arrow::Result<std::shared_ptr<S3ClientHolder>> AddClient(std::shared_ptr<S3Client> client);

  void Finalize();
  std::shared_lock<std::shared_mutex> LockShared();

  protected:
  friend class S3ClientHolder;

  std::shared_mutex mutex_;
  ClientHolderList holders_;
  bool finalized_ = false;
};

class ClientBuilder {
  public:
  explicit ClientBuilder(arrow::fs::S3Options options);

  arrow::Result<std::shared_ptr<S3ClientHolder>> BuildClient(
      std::optional<arrow::io::IOContext> io_context = std::nullopt);

  const Aws::Client::ClientConfiguration& config() const;
  Aws::Client::ClientConfiguration* mutable_config();
  const arrow::fs::S3Options& options() const;

  protected:
  arrow::fs::S3Options options_;
#ifdef ARROW_S3_HAS_S3CLIENT_CONFIGURATION
  Aws::S3::S3ClientConfiguration client_config_;
#else
  Aws::Client::ClientConfiguration client_config_;
#endif
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider_;
};

// singleton resource holder
std::shared_ptr<S3ClientFinalizer> GetClientFinalizer();

}  // namespace milvus_storage
