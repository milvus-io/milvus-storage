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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <arrow/io/file.h>

#include "milvus-storage/common/extend_status.h"
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

class DurableLocalOutput final : public arrow::io::OutputStream {
  public:
  DurableLocalOutput(std::shared_ptr<arrow::io::FileOutputStream> stream,
                     std::string path,
                     std::string temporary,
                     bool exclusive)
      : stream_(std::move(stream)), path_(std::move(path)), temporary_(std::move(temporary)), exclusive_(exclusive) {}

  ~DurableLocalOutput() override { (void)Abort(); }

  bool closed() const override { return closed_; }

  arrow::Result<int64_t> Tell() const override {
    if (closed_)
      return arrow::Status::Invalid("Output stream is closed: ", path_);
    return stream_->Tell();
  }

  arrow::Status Write(const void* data, int64_t size) override {
    if (closed_)
      return arrow::Status::Invalid("Output stream is closed: ", path_);
    if (!status_.ok())
      return status_;
    status_ = stream_->Write(data, size);
    return status_;
  }

  arrow::Status Flush() override {
    if (closed_)
      return arrow::Status::Invalid("Output stream is closed: ", path_);
    if (!status_.ok())
      return status_;
    status_ = stream_->Flush();
    return status_;
  }

  arrow::Status Abort() override {
    if (closed_)
      return arrow::Status::OK();
    closed_ = true;
    status_ = arrow::Status::Cancelled("Local object publication aborted: ", path_);
    auto status = stream_->Close();
    if (!temporary_.empty())
      (void)::unlink(temporary_.c_str());
    return status;
  }

  arrow::Status Close() override {
    if (closed_)
      return status_;
    if (status_.ok())
      status_ = SyncFD(stream_->file_descriptor(), path_, false);
    FIU_DO_ON(FIUKEY_LOCAL_OUTPUT_CLOSE_FAIL, {
      if (status_.ok())
        status_ = arrow::Status::IOError("Injected local close failure");
    });
    auto close_status = stream_->Close();
    closed_ = true;
    if (status_.ok())
      status_ = close_status;
    if (!status_.ok()) {
      if (!temporary_.empty())
        (void)::unlink(temporary_.c_str());
      return status_;
    }
    if (!temporary_.empty()) {
      // link is an atomic no-replace publication. A check-then-open guarded by
      // a process mutex cannot provide manifest CAS or hide incomplete files.
      int result = exclusive_ ? ::link(temporary_.c_str(), path_.c_str()) : ::rename(temporary_.c_str(), path_.c_str());
      const int error = errno;
      if (exclusive_ || result != 0)
        (void)::unlink(temporary_.c_str());
      temporary_.clear();
      if (result != 0) {
        status_ = exclusive_ && error == EEXIST
                      ? MakeExtendError(ExtendStatusCode::AwsErrorConflict, "File already exists: " + path_, "")
                      : IOError("publish", path_, error);
        return status_;
      }
    }
    status_ = SyncLocalDirectoryTree(std::filesystem::path(path_).parent_path().string());
    return status_;
  }

  private:
  std::shared_ptr<arrow::io::FileOutputStream> stream_;
  std::string path_;
  std::string temporary_;
  bool exclusive_;
  bool closed_ = false;
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

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenDurableLocalOutput(const std::string& path,
                                                                               bool exclusive,
                                                                               bool append) {
  static std::atomic<uint64_t> sequence{0};
  std::string temporary;
  int fd = -1;
  struct stat existing;
  bool preserve_mode = !append && !exclusive && ::stat(path.c_str(), &existing) == 0 && S_ISREG(existing.st_mode);
  if (append) {
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
  } else {
    const auto parent = std::filesystem::path(path).parent_path();
    for (int attempt = 0; attempt < 16; ++attempt) {
      temporary = (parent / (".milvus-tmp-" + std::to_string(::getpid()) + "-" +
                             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                             std::to_string(sequence.fetch_add(1))))
                      .string();
      fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
      if (fd >= 0 || errno != EEXIST)
        break;
    }
  }
  if (fd < 0)
    return IOError("open output", path, errno);
  // Replacement must not reset permissions on an existing object. The initial
  // open still applies the process umask to newly created objects.
  if (preserve_mode && ::fchmod(fd, existing.st_mode & 0777) != 0) {
    const int error = errno;
    (void)::close(fd);
    (void)::unlink(temporary.c_str());
    return IOError("preserve permissions", path, error);
  }
  auto stream = arrow::io::FileOutputStream::Open(fd);
  if (!stream.ok()) {
    (void)::close(fd);
    if (!temporary.empty())
      (void)::unlink(temporary.c_str());
    return stream.status();
  }
  return std::make_shared<DurableLocalOutput>(std::move(stream).ValueOrDie(), path, temporary, exclusive);
}

}  // namespace milvus_storage
