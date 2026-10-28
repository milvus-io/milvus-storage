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
  SyncOnCloseOutput(std::shared_ptr<arrow::io::FileOutputStream> stream, std::string path, std::string root)
      : stream_(std::move(stream)), path_(std::move(path)), root_(std::move(root)) {}

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
      status_ = SyncLocalDirectoryTree(std::filesystem::path(path_).parent_path().string(), root_);
    return status_;
  }

  private:
  std::shared_ptr<arrow::io::FileOutputStream> stream_;
  std::string path_;
  std::string root_;
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

arrow::Status SyncLocalDirectoryTree(const std::string& directory, const std::string& root) {
  std::error_code error;
  // canonical resolves symlinks before '..', matching POSIX I/O. Lexical
  // normalization can silently select an unrelated directory for fsync.
  auto current = std::filesystem::canonical(directory.empty() ? "." : directory, error);
  if (error)
    return arrow::Status::IOError("resolve directory ", directory, ": ", error.message());
  auto boundary = std::filesystem::canonical(root, error);
  if (error)
    return arrow::Status::IOError("resolve storage root ", root, ": ", error.message());
  auto relative = current.lexically_relative(boundary);
  if (relative.empty() || *relative.begin() == "..")
    return arrow::Status::Invalid("Directory ", directory, " is outside durability root ", root);
  for (;;) {
    int fd = ::open(current.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0)
      return IOError("open directory", current.string(), errno);
    auto status = SyncFD(fd, current.string(), true);
    if (::close(fd) != 0 && status.ok())
      status = IOError("close directory", current.string(), errno);
    ARROW_RETURN_NOT_OK(status);
    if (current == boundary)
      return arrow::Status::OK();
    current = current.parent_path();
  }
}

arrow::Result<std::string> PrepareLocalRoot(const std::string& root) {
  std::error_code error;
  auto path = std::filesystem::absolute(root.empty() ? "." : root, error);
  if (error)
    return arrow::Status::IOError("resolve storage root ", root, ": ", error.message());
  auto ancestor = path;
  while (!std::filesystem::exists(ancestor, error)) {
    if (error)
      return arrow::Status::IOError("stat storage root ", ancestor.string(), ": ", error.message());
    ancestor = ancestor.parent_path();
  }
  std::filesystem::create_directories(path, error);
  if (error)
    return arrow::Status::IOError("create storage root ", root, ": ", error.message());
  if (ancestor != path)
    ARROW_RETURN_NOT_OK(SyncLocalDirectoryTree(path.string(), ancestor.string()));
  auto resolved = std::filesystem::canonical(path, error);
  if (error)
    return arrow::Status::IOError("resolve storage root ", root, ": ", error.message());
  return resolved.string();
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenDurableLocalOutput(const std::string& path,
                                                                               bool append,
                                                                               const std::string& root) {
  ARROW_ASSIGN_OR_RAISE(auto stream, arrow::io::FileOutputStream::Open(path, append));
  return std::make_shared<SyncOnCloseOutput>(std::move(stream), path, root);
}

}  // namespace milvus_storage
