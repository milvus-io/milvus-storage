// Copyright 2023 Zilliz
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

#include <arrow/filesystem/filesystem.h>
#include <arrow/util/macros.h>
#include <arrow/util/uri.h>
#include "opendal.h"

namespace milvus_storage {

class OpendalOptions {
  public:
  static arrow::Result<OpendalOptions> FromUri(const arrow::internal::Uri& uri, std::string* out_path);

  const std::unordered_map<std::string, std::string>& options() const { return options_; }

  const std::string& at(const std::string& key) const { return options_.at(key); }

  protected:
  std::unordered_map<std::string, std::string> options_;
};

class OpendalFileSystem : public arrow::fs::FileSystem {
  public:
  ~OpendalFileSystem() override;

  std::string type_name() const override { return "opendal"; }

  bool Equals(const FileSystem& other) const override;

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override;
  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& select) override {
    throw std::runtime_error("Not implemented");
  };

  arrow::Status CreateDir(const std::string& path, bool recursive = true) override;

  arrow::Status DeleteDir(const std::string& path) override;
  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok = false) override;
  arrow::Status DeleteRootDirContents() override { throw std::runtime_error("Not implemented"); }

  arrow::Status DeleteFile(const std::string& path) override;

  arrow::Status Move(const std::string& src, const std::string& dest) override;

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override;

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata = {}) override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata = {}) override;

  /// Create a S3FileSystem instance from the given options.
  static arrow::Result<std::unique_ptr<OpendalFileSystem>> Make(
      const OpendalOptions& options, const arrow::io::IOContext& = arrow::io::default_io_context());

  protected:
  OpendalFileSystem(const OpendalOptions& options, const arrow::io::IOContext& io_context);
  opendal_operator* operator_;
  OpendalOptions options_;
};

}  // namespace milvus_storage
