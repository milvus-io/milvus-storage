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

#include <arrow/filesystem/localfs.h>

#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/upload_conditional.h"

namespace milvus_storage {

static constexpr auto local_uri_scheme = "file://";

/// \brief Wrapper for LocalFileSystem that implements Observable and UploadConditional
class LocalFileSystemWrapper : public arrow::fs::LocalFileSystem, public UploadConditional, public Observable {
  private:
  // Macros to simplify metrics tracking
#define TRACK_METRICS(counter, call)    \
  do {                                  \
    metrics_->counter();                \
    auto result = call;                 \
    if (!result.ok()) {                 \
      metrics_->IncrementFailedCount(); \
    }                                   \
    return result;                      \
  } while (0)

#define TRACK_METRICS_AND_WRAP(counter, call, WrapperType)                          \
  do {                                                                              \
    metrics_->counter();                                                            \
    auto result = call;                                                             \
    if (!result.ok()) {                                                             \
      metrics_->IncrementFailedCount();                                             \
      return result.status();                                                       \
    }                                                                               \
    return std::make_shared<WrapperType>(std::move(result.ValueOrDie()), metrics_); \
  } while (0)

  public:
  explicit LocalFileSystemWrapper(const arrow::fs::LocalFileSystemOptions& options)
      : arrow::fs::LocalFileSystem(options), metrics_(std::make_shared<FilesystemMetrics>()) {}

  std::shared_ptr<FilesystemMetrics> GetMetrics() const override { return metrics_; }

  // Override methods to track metrics
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    TRACK_METRICS(IncrementGetFileInfoCount, arrow::fs::LocalFileSystem::GetFileInfo(path));
  }

  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override {
    TRACK_METRICS(IncrementGetFileInfoCount, arrow::fs::LocalFileSystem::GetFileInfo(select));
  }

  arrow::Status CreateDir(const std::string& path, bool recursive) override {
    TRACK_METRICS(IncrementCreateDirCount, arrow::fs::LocalFileSystem::CreateDir(path, recursive));
  }

  arrow::Status DeleteDir(const std::string& path) override {
    TRACK_METRICS(IncrementDeleteDirCount, arrow::fs::LocalFileSystem::DeleteDir(path));
  }

  arrow::Status DeleteFile(const std::string& path) override {
    TRACK_METRICS(IncrementDeleteFileCount, arrow::fs::LocalFileSystem::DeleteFile(path));
  }

  arrow::Status Move(const std::string& src, const std::string& dest) override {
    TRACK_METRICS(IncrementMoveCount, arrow::fs::LocalFileSystem::Move(src, dest));
  }

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override {
    TRACK_METRICS(IncrementCopyFileCount, arrow::fs::LocalFileSystem::CopyFile(src, dest));
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override {
    TRACK_METRICS_AND_WRAP(IncrementReadCount, arrow::fs::LocalFileSystem::OpenInputStream(path), MetricsInputStream);
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    TRACK_METRICS_AND_WRAP(IncrementReadCount, arrow::fs::LocalFileSystem::OpenInputStream(info.path()),
                           MetricsInputStream);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    TRACK_METRICS_AND_WRAP(IncrementReadCount, arrow::fs::LocalFileSystem::OpenInputFile(path),
                           MetricsRandomAccessFile);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    TRACK_METRICS_AND_WRAP(IncrementReadCount, arrow::fs::LocalFileSystem::OpenInputFile(info.path()),
                           MetricsRandomAccessFile);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    TRACK_METRICS_AND_WRAP(IncrementWriteCount, arrow::fs::LocalFileSystem::OpenOutputStream(path, metadata),
                           MetricsOutputStream);
  }

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
    TRACK_METRICS(IncrementDeleteDirCount, arrow::fs::LocalFileSystem::DeleteDirContents(path, missing_dir_ok));
  }

  arrow::Status DeleteRootDirContents() override {
    TRACK_METRICS(IncrementDeleteDirCount, arrow::fs::LocalFileSystem::DeleteRootDirContents());
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    TRACK_METRICS(IncrementWriteCount, arrow::fs::LocalFileSystem::OpenAppendStream(path, metadata));
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) override {
    // Check if file already exists, this is NOT thread safe.
    auto file_info_result = arrow::fs::LocalFileSystem::GetFileInfo(path);
    if (!file_info_result.ok()) {
      metrics_->IncrementFailedCount();
      return file_info_result.status();
    }
    auto file_info = file_info_result.ValueOrDie();
    if (file_info.type() == arrow::fs::FileType::File) {
      metrics_->IncrementFailedCount();
      return arrow::Status::IOError("File already exists: ", path);
    }
    return OpenOutputStream(path, metadata);
  }

  private:
  std::shared_ptr<FilesystemMetrics> metrics_;

#undef TRACK_METRICS
#undef TRACK_METRICS_AND_WRAP
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
