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
#include "common/log.h"
#include "common/macro.h"

#include <arrow/util/key_value_metadata.h>
#include <arrow/filesystem/s3fs.h>
#include "arrow/filesystem/filesystem.h"
#include "arrow/util/macros.h"
#include "arrow/util/uri.h"
#include "arrow/io/interfaces.h"

using namespace arrow;
using ::arrow::fs::FileInfo;

namespace milvus_storage {

class MultiPartUploadS3FS : public arrow::fs::S3FileSystem {
  public:
  ~MultiPartUploadS3FS() override;

  std::string type_name() const override { return "multiPartUploadS3"; }

  bool Equals(const FileSystem& other) const override;

  arrow::Result<std::string> PathFromUri(const std::string& uri_string) const override;

  // arrow::Result<FileInfo> GetFileInfo(const std::string& path) override;

  // arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override;

  // FileInfoGenerator GetFileInfoGenerator(const FileSelector& select) override;

  // arrow::Status CreateDir(const std::string& path, bool recursive) override;

  arrow::Status DeleteDir(const std::string& path) override;
  // Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override;
  // Future<> DeleteDirContentsAsync(const std::string& path, bool missing_dir_ok) override;
  // Status DeleteRootDirContents() override;

  // Status DeleteFile(const std::string& path) override;

  // Status Move(const std::string& src, const std::string& dest) override;

  // Status CopyFile(const std::string& src, const std::string& dest) override;

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

}  // namespace milvus_storage