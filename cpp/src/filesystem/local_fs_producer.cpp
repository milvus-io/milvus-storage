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

#include <mutex>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <arrow/filesystem/localfs.h>

#include "milvus-storage/filesystem/observable.h"
#include "milvus-storage/filesystem/upload_conditional.h"
#include "milvus-storage/common/extend_status.h"

namespace milvus_storage {

static constexpr auto local_uri_scheme = "file://";

/// \brief Wrapper for LocalFileSystem that implements Observable and UploadConditional
class LocalFileSystemWrapper : public arrow::fs::LocalFileSystem, public UploadConditional, public Observable {
  public:
  explicit LocalFileSystemWrapper(const arrow::fs::LocalFileSystemOptions& options)
      : arrow::fs::LocalFileSystem(options), metrics_(std::make_shared<FilesystemMetrics>()) {}

  std::shared_ptr<FilesystemMetrics> GetMetrics() const override { return metrics_; }

  // Override methods to track metrics
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override {
    auto op = metrics_->StartOp(OpType::Head);
    auto result = arrow::fs::LocalFileSystem::GetFileInfo(path);
    if (!result.ok())
      op.Fail(ClassifyArrowStatus(result.status()));
    return result;
  }

  arrow::Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(const arrow::fs::FileSelector& select) override {
    auto op = metrics_->StartOp(OpType::List);
    auto result = arrow::fs::LocalFileSystem::GetFileInfo(select);
    if (!result.ok())
      op.Fail(ClassifyArrowStatus(result.status()));
    return result;
  }

  arrow::Status CreateDir(const std::string& path, bool recursive) override {
    auto op = metrics_->StartOp(OpType::CreateDir);
    auto st = arrow::fs::LocalFileSystem::CreateDir(path, recursive);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Status DeleteDir(const std::string& path) override {
    auto op = metrics_->StartOp(OpType::DeleteDir);
    auto st = arrow::fs::LocalFileSystem::DeleteDir(path);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Status DeleteFile(const std::string& path) override {
    auto op = metrics_->StartOp(OpType::DeleteFile);
    auto st = arrow::fs::LocalFileSystem::DeleteFile(path);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Status Move(const std::string& src, const std::string& dest) override {
    auto op = metrics_->StartOp(OpType::Move);
    auto st = arrow::fs::LocalFileSystem::Move(src, dest);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Status CopyFile(const std::string& src, const std::string& dest) override {
    auto op = metrics_->StartOp(OpType::Copy);
    auto st = arrow::fs::LocalFileSystem::CopyFile(src, dest);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const std::string& path) override {
    auto op = metrics_->StartOp(OpType::OpenInput);
    auto result = arrow::fs::LocalFileSystem::OpenInputStream(path);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result.status();
    }
    return std::make_shared<MetricsInputStream>(std::move(result.ValueOrDie()), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::InputStream>> OpenInputStream(const arrow::fs::FileInfo& info) override {
    auto op = metrics_->StartOp(OpType::OpenInput);
    auto result = arrow::fs::LocalFileSystem::OpenInputStream(info.path());
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result.status();
    }
    return std::make_shared<MetricsInputStream>(std::move(result.ValueOrDie()), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    auto op = metrics_->StartOp(OpType::OpenInput);
    auto result = arrow::fs::LocalFileSystem::OpenInputFile(path);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result.status();
    }
    return std::make_shared<MetricsRandomAccessFile>(std::move(result.ValueOrDie()), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    auto op = metrics_->StartOp(OpType::OpenInput);
    auto result = arrow::fs::LocalFileSystem::OpenInputFile(info.path());
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result.status();
    }
    return std::make_shared<MetricsRandomAccessFile>(std::move(result.ValueOrDie()), metrics_);
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    auto op = metrics_->StartOp(OpType::OpenOutput);
    auto result = arrow::fs::LocalFileSystem::OpenOutputStream(path, metadata);
    if (!result.ok()) {
      op.Fail(ClassifyArrowStatus(result.status()));
      return result.status();
    }
    return std::make_shared<MetricsOutputStream>(std::move(result.ValueOrDie()), metrics_);
  }

  arrow::Status DeleteDirContents(const std::string& path, bool missing_dir_ok) override {
    auto op = metrics_->StartOp(OpType::DeleteDir);
    auto st = arrow::fs::LocalFileSystem::DeleteDirContents(path, missing_dir_ok);
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Status DeleteRootDirContents() override {
    auto op = metrics_->StartOp(OpType::DeleteDir);
    auto st = arrow::fs::LocalFileSystem::DeleteRootDirContents();
    if (!st.ok())
      op.Fail(ClassifyArrowStatus(st));
    return st;
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenAppendStream(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override {
    auto op = metrics_->StartOp(OpType::OpenOutput);
    auto result = arrow::fs::LocalFileSystem::OpenAppendStream(path, metadata);
    if (!result.ok())
      op.Fail(ClassifyArrowStatus(result.status()));
    return result;
  }

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenConditionalOutputStream(
      const std::string& path, std::shared_ptr<arrow::KeyValueMetadata> metadata) override {
    // This lock is only for testing purposes.
    static std::mutex local_conditional_write_mutex;
    std::scoped_lock lock(local_conditional_write_mutex);

    arrow::Result<arrow::fs::FileInfo> file_info_result = [&] {
      auto op = metrics_->StartOp(OpType::Head);
      auto result = arrow::fs::LocalFileSystem::GetFileInfo(path);
      if (!result.ok())
        op.Fail(ClassifyArrowStatus(result.status()));
      return result;
    }();
    if (!file_info_result.ok()) {
      return file_info_result.status();
    }
    auto file_info = file_info_result.ValueOrDie();
    if (file_info.type() == arrow::fs::FileType::File) {
      return MakeExtendError(ExtendStatusCode::AwsErrorConflict, "File already exists: " + path, "");
    }
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
