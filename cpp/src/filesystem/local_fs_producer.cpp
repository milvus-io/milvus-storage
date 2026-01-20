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

#include "milvus-storage/filesystem/local_fs_producer.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <arrow/buffer.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/uri.h>

#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/upload_conditional.h"

namespace milvus_storage {

static constexpr auto local_uri_scheme = "file://";

/// \brief Wrapper for InputStream to track read bytes
class MetricsInputStream : public arrow::io::InputStream {
  public:
  MetricsInputStream(std::shared_ptr<arrow::io::InputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, stream_->Read(nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, stream_->Read(nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Status Close() override { return stream_->Close(); }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }

  private:
  std::shared_ptr<arrow::io::InputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for RandomAccessFile to track read bytes
class MetricsRandomAccessFile : public arrow::io::RandomAccessFile {
  public:
  MetricsRandomAccessFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::shared_ptr<FilesystemMetrics> metrics)
      : file_(std::move(file)), metrics_(std::move(metrics)) {}

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, file_->Read(nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, file_->Read(nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    ARROW_ASSIGN_OR_RAISE(auto bytes_read, file_->ReadAt(position, nbytes, out));
    if (bytes_read > 0) {
      metrics_->IncrementReadBytes(bytes_read);
    }
    return bytes_read;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, file_->ReadAt(position, nbytes));
    if (buffer && buffer->size() > 0) {
      metrics_->IncrementReadBytes(buffer->size());
    }
    return buffer;
  }

  arrow::Status Close() override { return file_->Close(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for OutputStream to track write bytes
class MetricsOutputStream : public arrow::io::OutputStream {
  public:
  MetricsOutputStream(std::shared_ptr<arrow::io::OutputStream> stream, std::shared_ptr<FilesystemMetrics> metrics)
      : stream_(std::move(stream)), metrics_(std::move(metrics)) {}

  arrow::Status Write(const void* data, int64_t nbytes) override {
    auto status = stream_->Write(data, nbytes);
    if (status.ok() && nbytes > 0) {
      metrics_->IncrementWriteBytes(nbytes);
    }
    return status;
  }

  arrow::Status Close() override { return stream_->Close(); }
  bool closed() const override { return stream_->closed(); }
  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }
  arrow::Status Flush() override { return stream_->Flush(); }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
  std::shared_ptr<FilesystemMetrics> metrics_;
};

/// \brief Wrapper for LocalFileSystem that implements Observable and UploadConditional
class LocalFileSystemWrapper : public arrow::fs::LocalFileSystem, public UploadConditional, public Observable {
  public:
  explicit LocalFileSystemWrapper(const arrow::fs::LocalFileSystemOptions& options)
      : arrow::fs::LocalFileSystem(options), metrics_(std::make_shared<FilesystemMetrics>()) {}

  std::shared_ptr<FilesystemMetrics> GetMetrics() const override { return metrics_; }

  // Override methods to track metrics
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    metrics_->IncrementGetFileInfoCount();
    auto result = arrow::fs::LocalFileSystem::GetFileInfo(path);
    if (!result.ok()) {
      metrics_->IncrementFailedCount();
    }
    return result;
  }

  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override {
    metrics_->IncrementGetFileInfoCount();
    auto result = arrow::fs::LocalFileSystem::GetFileInfo(select);
    if (!result.ok()) {
      metrics_->IncrementFailedCount();
    }
    return result;
  }

  arrow::Status CreateDir(const std::string& path, bool recursive) override {
    metrics_->IncrementCreateDirCount();
    auto status = arrow::fs::LocalFileSystem::CreateDir(path, recursive);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Status DeleteDir(const std::string& path) override {
    metrics_->IncrementDeleteDirCount();
    auto status = arrow::fs::LocalFileSystem::DeleteDir(path);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Status DeleteFile(const std::string& path) override {
    metrics_->IncrementDeleteFileCount();
    auto status = arrow::fs::LocalFileSystem::DeleteFile(path);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Status Move(const std::string& src, const std::string& dest) override {
    metrics_->IncrementMoveCount();
    auto status = arrow::fs::LocalFileSystem::Move(src, dest);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override {
    metrics_->IncrementCopyFileCount();
    auto status = arrow::fs::LocalFileSystem::CopyFile(src, dest);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override {
    metrics_->IncrementReadCount();
    ARROW_ASSIGN_OR_RAISE(auto stream, arrow::fs::LocalFileSystem::OpenInputStream(path));
    return std::make_shared<MetricsInputStream>(std::move(stream), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    metrics_->IncrementReadCount();
    ARROW_ASSIGN_OR_RAISE(auto stream, arrow::fs::LocalFileSystem::OpenInputStream(info.path()));
    return std::make_shared<MetricsInputStream>(std::move(stream), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    metrics_->IncrementReadCount();
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::LocalFileSystem::OpenInputFile(path));
    return std::make_shared<MetricsRandomAccessFile>(std::move(file), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    metrics_->IncrementReadCount();
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::fs::LocalFileSystem::OpenInputFile(info.path()));
    return std::make_shared<MetricsRandomAccessFile>(std::move(file), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    metrics_->IncrementWriteCount();
    ARROW_ASSIGN_OR_RAISE(auto stream, arrow::fs::LocalFileSystem::OpenOutputStream(path, metadata));
    return std::make_shared<MetricsOutputStream>(std::move(stream), metrics_);
  }

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
    metrics_->IncrementDeleteDirCount();
    auto status = arrow::fs::LocalFileSystem::DeleteDirContents(path, missing_dir_ok);
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Status DeleteRootDirContents() override {
    metrics_->IncrementDeleteDirCount();
    auto status = arrow::fs::LocalFileSystem::DeleteRootDirContents();
    if (!status.ok()) {
      metrics_->IncrementFailedCount();
    }
    return status;
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    metrics_->IncrementWriteCount();
    auto result = arrow::fs::LocalFileSystem::OpenAppendStream(path, metadata);
    if (!result.ok()) {
      metrics_->IncrementFailedCount();
    }
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) override {
    // Check if file already exists using base class to avoid double-counting GetFileInfo
    ARROW_ASSIGN_OR_RAISE(auto file_info, arrow::fs::LocalFileSystem::GetFileInfo(path));
    if (file_info.type() == arrow::fs::FileType::File) {
      metrics_->IncrementFailedCount();
      return arrow::Status::IOError("File already exists: ", path);
    }
    // File doesn't exist (NotFound or Directory), proceed with normal output stream
    return OpenOutputStream(path, metadata);
  }

  private:
  std::shared_ptr<FilesystemMetrics> metrics_;
};

arrow::Result<ArrowFileSystemPtr> LocalFileSystemProducer::Make() {
  std::string out_path;
  auto path = boost::filesystem::path(config_.root_path);
  if (path.is_relative()) {
    path = boost::filesystem::absolute(path);
  }
  std::string local_uri = local_uri_scheme + path.string();

  ARROW_ASSIGN_OR_RAISE(auto arrow_uri, arrow::util::Uri::FromString(local_uri));
  ARROW_ASSIGN_OR_RAISE(auto option, arrow::fs::LocalFileSystemOptions::FromUri(arrow_uri, &out_path));

  // create local dir if not exists
  // if exists, check it is a directory
  boost::filesystem::path dir_path(out_path);
  if (!boost::filesystem::exists(dir_path)) {
    boost::filesystem::create_directories(dir_path);
  } else if (!boost::filesystem::is_directory(dir_path)) {
    return arrow::Status::Invalid("Path ", out_path, " is not a directory");
  }

  return std::make_shared<FileSystemProxy>(out_path, std::make_shared<LocalFileSystemWrapper>(option));
}

}  // namespace milvus_storage
