// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/local_durability.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

#include <arrow/io/file.h>

#include "milvus-storage/common/fiu_local.h"

namespace milvus_storage {
namespace {

arrow::Status IOError(const std::string& operation, const std::string& path, int error) {
  return arrow::Status::IOError(operation, " ", path, ": ", std::strerror(error));
}

arrow::Status SyncFD(int fd, const std::string& path, bool directory) {
  if (directory) {
    FIU_RETURN_ON(FIUKEY_LOCAL_DIRECTORY_SYNC_FAIL, arrow::Status::IOError("Injected local directory sync failure"));
  } else {
    FIU_RETURN_ON(FIUKEY_LOCAL_FILE_SYNC_FAIL, arrow::Status::IOError("Injected local file sync failure"));
  }
  int result;
  do {
    result = ::fsync(fd);
  } while (result < 0 && errno == EINTR);
  return result == 0 ? arrow::Status::OK() : IOError("fsync", path, errno);
}

// Keep the filesystem's existing direct-write behavior. A failed or abandoned
// write can leave an incomplete file; callers must not commit its metadata.
class SyncOnCloseOutput final : public arrow::io::OutputStream {
  public:
  SyncOnCloseOutput(std::shared_ptr<arrow::io::FileOutputStream> stream, std::string path)
      : stream_(std::move(stream)), path_(std::move(path)) {}

  ~SyncOnCloseOutput() override { (void)Abort(); }

  bool closed() const override { return stream_->closed(); }

  arrow::Result<int64_t> Tell() const override { return stream_->Tell(); }

  arrow::Status Write(const void* data, int64_t size) override {
    if (!status_.ok())
      return status_;
    status_ = stream_->Write(data, size);
    return status_;
  }

  arrow::Status Flush() override {
    if (!status_.ok())
      return status_;
    status_ = stream_->Flush();
    return status_;
  }

  arrow::Status Abort() override {
    if (closed())
      return arrow::Status::OK();
    status_ = arrow::Status::Cancelled("Local output aborted: ", path_);
    return stream_->Close();
  }

  arrow::Status Close() override {
    if (closed())
      return status_;
    if (status_.ok())
      status_ = SyncFD(stream_->file_descriptor(), path_, false);
    FIU_DO_ON(FIUKEY_LOCAL_OUTPUT_CLOSE_FAIL, {
      if (status_.ok())
        status_ = arrow::Status::IOError("Injected local close failure");
    });
    auto close_status = stream_->Close();
    if (status_.ok())
      status_ = close_status;
    if (status_.ok())
      status_ = SyncLocalDirectoryTree(std::filesystem::path(path_).parent_path().string());
    return status_;
  }

  private:
  std::shared_ptr<arrow::io::FileOutputStream> stream_;
  std::string path_;
  arrow::Status status_;
};

}  // namespace

arrow::Status SyncLocalFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return IOError("open for sync", path, errno);
  auto status = SyncFD(fd, path, false);
  if (::close(fd) != 0 && status.ok())
    status = IOError("close", path, errno);
  return status;
}

arrow::Status SyncLocalDirectoryTree(const std::string& directory) {
  std::error_code error;
  auto current = std::filesystem::absolute(directory.empty() ? "." : directory, error).lexically_normal();
  if (error)
    return arrow::Status::IOError("resolve directory ", directory, ": ", error.message());
  for (;;) {
    int fd = ::open(current.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0)
      return IOError("open directory", current.string(), errno);
    auto status = SyncFD(fd, current.string(), true);
    if (::close(fd) != 0 && status.ok())
      status = IOError("close directory", current.string(), errno);
    ARROW_RETURN_NOT_OK(status);
    auto parent = current.parent_path();
    if (parent == current)
      return arrow::Status::OK();
    current = std::move(parent);
  }
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenDurableLocalOutput(const std::string& path, bool append) {
  ARROW_ASSIGN_OR_RAISE(auto stream, arrow::io::FileOutputStream::Open(path, append));
  return std::make_shared<SyncOnCloseOutput>(std::move(stream), path);
}

}  // namespace milvus_storage
