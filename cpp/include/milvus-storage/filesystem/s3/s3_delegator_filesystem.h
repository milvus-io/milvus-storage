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
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>

#define kMultiPartUploadFileSystemType "multipart-upload"
#define kMultiPartUploadSizeKey "multi_part_upload_size"

namespace milvus_storage {

/// \brief A FileSystem implementation that delegates to another
/// implementation after do metrics or provider private function.
class S3DelegatorFileSystem : public arrow::fs::FileSystem {
  public:
  // This constructor may abort if base_path is invalid.
  explicit S3DelegatorFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs);
  ~S3DelegatorFileSystem() override = default;

  std::string type_name() const override { return kMultiPartUploadFileSystemType; }
  std::shared_ptr<arrow::fs::FileSystem> base_fs() const { return base_fs_; }

  arrow::Result<std::string> NormalizePath(std::string path) override;
  arrow::Result<std::string> PathFromUri(const std::string& uri_string) const override;

  bool Equals(const arrow::fs::FileSystem& other) const override;

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  arrow::Result<arrow::fs::FileInfoVector> GetFileInfo(const arrow::fs::FileSelector& select) override;

  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& select) override;

  arrow::Status CreateDir(const std::string& path, bool recursive) override;

  arrow::Status DeleteDir(const std::string& path) override;
  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override;
  arrow::Status DeleteRootDirContents() override;

  arrow::Status DeleteFile(const std::string& path) override;

  arrow::Status Move(const std::string& src, const std::string& dest) override;

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override;

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;
  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override;

  arrow::Future<std::shared_ptr<arrow::io::InputStream>> OpenInputStreamAsync(const std::string& path) override;
  arrow::Future<std::shared_ptr<arrow::io::InputStream>> OpenInputStreamAsync(const arrow::fs::FileInfo& info) override;
  arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFileAsync(const std::string& path) override;
  arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFileAsync(
      const arrow::fs::FileInfo& info) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamWithUploadSize(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata, const int64_t part_size);

  protected:
  S3DelegatorFileSystem() = default;
  std::shared_ptr<arrow::fs::FileSystem> base_fs_;
};

}  // namespace milvus_storage
