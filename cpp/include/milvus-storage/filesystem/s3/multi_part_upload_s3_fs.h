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

#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

#include <aws/core/Aws.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>

#include <arrow/util/key_value_metadata.h>
#include "arrow/filesystem/filesystem.h"
#include "arrow/io/interfaces.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/filesystem/s3/s3_options.h"

using ::arrow::fs::FileInfo;
using ::arrow::fs::FileInfoGenerator;

namespace milvus_storage {

class S3ClientMetrics {
  public:
  S3ClientMetrics() = default;
  ~S3ClientMetrics() = default;

  void IncrementMultiPartUploadCreated() { multi_part_upload_created.fetch_add(1, std::memory_order_relaxed); }
  void IncrementMultiPartUploadFinished() { multi_part_upload_finished.fetch_add(1, std::memory_order_relaxed); }
  void IncrementUploadBytes(int64_t bytes) { upload_bytes.fetch_add(bytes, std::memory_order_relaxed); }
  void IncrementDownloadBytes(int64_t bytes) { download_bytes.fetch_add(bytes, std::memory_order_relaxed); }
  void IncrementUploadCount() { upload_count.fetch_add(1, std::memory_order_relaxed); }
  void IncrementDownloadCount() { download_count.fetch_add(1, std::memory_order_relaxed); }
  void IncrementFailedCount() { failed_count.fetch_add(1, std::memory_order_relaxed); }

  int64_t GetMultiPartUploadCreated() const { return multi_part_upload_created.load(std::memory_order_relaxed); }
  int64_t GetMultiPartUploadFinished() const { return multi_part_upload_finished.load(std::memory_order_relaxed); }
  int64_t GetUploadCount() const { return upload_count.load(std::memory_order_relaxed); }
  int64_t GetDownloadCount() const { return download_count.load(std::memory_order_relaxed); }
  int64_t GetUploadBytes() const { return upload_bytes.load(std::memory_order_relaxed); }
  int64_t GetDownloadBytes() const { return download_bytes.load(std::memory_order_relaxed); }
  int64_t GetFailedCount() const { return failed_count.load(std::memory_order_relaxed); }

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

class MultiPartUploadS3FS : public arrow::fs::FileSystem {
  public:
  ~MultiPartUploadS3FS() override;

  std::string type_name() const override { return MULTI_PART_UPLOAD_S3_FILESYSTEM_NAME; }

  /// Return the original S3 options when constructing the filesystem
  S3Options options() const;
  /// Return the actual region this filesystem connects to
  std::string region() const;

  bool Equals(const FileSystem& other) const override;

  arrow::Result<std::string> PathFromUri(const std::string& uri_string) const override;

  arrow::Result<FileInfo> GetFileInfo(const std::string& path) override;

  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override;

  FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& select) override;

  arrow::Status CreateDir(const std::string& path, bool recursive) override;

  arrow::Status DeleteDir(const std::string& path) override;

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override;

  arrow::Future<> DeleteDirContentsAsync(const std::string& path, bool missing_dir_ok) override;

  arrow::Status DeleteRootDirContents() override;

  arrow::Status DeleteFile(const std::string& path) override;

  arrow::Status Move(const std::string& src, const std::string& dest) override;

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(const std::string& s,
                                                                                         int64_t part_size);

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& s, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, int64_t part_size);

  static arrow::Result<std::shared_ptr<MultiPartUploadS3FS>> Make(
      const S3Options& options, const arrow::io::IOContext& = arrow::io::default_io_context());

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const FileInfo& info) override;

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& s) override;

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const FileInfo& info) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<S3ClientMetrics>> GetMetrics();

  protected:
  explicit MultiPartUploadS3FS(const S3Options& options, const arrow::io::IOContext& io_context);

  class Impl;
  std::shared_ptr<Impl> impl_;
};

/// \brief Initialize the S3 APIs with the specified set of options.
///
/// It is required to call this function at least once before using S3FileSystem.
///
/// Once this function is called you MUST call FinalizeS3 before the end of the
/// application in order to avoid a segmentation fault at shutdown.
arrow::Status InitializeS3(const S3GlobalOptions& options);

/// \brief Ensure the S3 APIs are initialized, but only if not already done.
///
/// If necessary, this will call InitializeS3() with some default options.
arrow::Status EnsureS3Initialized();

/// Whether S3 was initialized, and not finalized.
bool IsS3Initialized();

/// Whether S3 was finalized.
bool IsS3Finalized();

/// \brief Check if S3 is initialized and return an error if not.
///
/// This function checks if S3 has been initialized and returns an appropriate
/// error status if it has not been initialized or has been finalized.
arrow::Status CheckS3Initialized();

/// \brief Shutdown the S3 APIs.
///
/// This can wait for some S3 concurrent calls to finish so as to avoid
/// race conditions.
/// After this function has been called, all S3 calls will fail with an error.
///
/// Calls to InitializeS3() and FinalizeS3() should be serialized by the
/// application (this also applies to EnsureS3Initialized() and
/// EnsureS3Finalized()).
arrow::Status FinalizeS3();

/// \brief Ensure the S3 APIs are shutdown, but only if not already done.
///
/// If necessary, this will call FinalizeS3().
arrow::Status EnsureS3Finalized();

}  // namespace milvus_storage
