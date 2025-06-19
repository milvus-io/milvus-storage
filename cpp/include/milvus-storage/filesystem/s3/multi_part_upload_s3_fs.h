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

#include <arrow/util/key_value_metadata.h>
#include <arrow/filesystem/s3fs.h>
#include "arrow/filesystem/filesystem.h"
#include "arrow/io/interfaces.h"

using ::arrow::fs::FileInfo;
using ::arrow::fs::FileInfoGenerator;

namespace milvus_storage {

class MultiPartUploadS3FS : public arrow::fs::S3FileSystem {
  public:
  ~MultiPartUploadS3FS() override;

  std::string type_name() const override { return "multiPartUploadS3"; }

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
      const arrow::fs::S3Options& options, const arrow::io::IOContext& = arrow::io::default_io_context());

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const FileInfo& info) override;

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& s) override;

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const FileInfo& info) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  protected:
  explicit MultiPartUploadS3FS(const arrow::fs::S3Options& options, const arrow::io::IOContext& io_context);

  class Impl;
  std::shared_ptr<Impl> impl_;
};

struct S3GlobalOptions {
  arrow::fs::S3LogLevel log_level;
  /// The number of threads to configure when creating AWS' I/O event loop
  ///
  /// Defaults to 1 as recommended by AWS' doc when the # of connections is
  /// expected to be, at most, in the hundreds
  ///
  /// For more details see Aws::Crt::Io::EventLoopGroup
  int num_event_loop_threads = 1;

  /// AWS SDK wide options for http
  Aws::HttpOptions http_options;

  /// Override default http options
  bool override_default_http_options = false;

  /// \brief Initialize with default options
  ///
  /// For log_level, this method first tries to extract a suitable value from the
  /// environment variable ARROW_S3_LOG_LEVEL.
  static S3GlobalOptions Defaults();
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