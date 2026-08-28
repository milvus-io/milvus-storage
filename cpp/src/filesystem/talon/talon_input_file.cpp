// Copyright 2025 Zilliz
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

#include "milvus-storage/filesystem/talon/talon_input_file.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>  // arrow::fs::kNoSize

namespace milvus_storage::talon {

namespace {
using ::arrow::fs::kNoSize;

arrow::Future<int64_t> FailedReadFuture(arrow::Status status) {
  return arrow::Future<int64_t>::MakeFinished(arrow::Result<int64_t>(std::move(status)));
}
}  // namespace

TalonInputFile::TalonInputFile(
    std::shared_ptr<TalonClient> client, std::string uri, int64_t size, std::string version, arrow::MemoryPool* pool)
    : client_(std::move(client)), uri_(std::move(uri)), identity_(std::make_shared<Identity>(size)), pool_(pool) {
  if (!version.empty()) {
    identity_->version = std::move(version);
    identity_->has_version = true;
  }
}

arrow::Status TalonInputFile::CheckClosed() const {
  if (closed_) {
    return arrow::Status::Invalid("Operation on closed Talon input file");
  }
  return arrow::Status::OK();
}

arrow::Status TalonInputFile::EnsureSize() {
  if (identity_->content_length.load(std::memory_order_acquire) != kNoSize) {
    return arrow::Status::OK();
  }
  ARROW_ASSIGN_OR_RAISE(auto stat, client_->Stat(uri_));
  int64_t expected = kNoSize;
  identity_->content_length.compare_exchange_strong(expected, stat.size, std::memory_order_acq_rel,
                                                    std::memory_order_acquire);
  if (!stat.version.empty()) {
    std::lock_guard<std::mutex> lock(identity_->version_mutex);
    if (!identity_->has_version) {
      identity_->version = std::move(stat.version);
      identity_->has_version = true;
    }
  }
  return arrow::Status::OK();
}

arrow::Result<int64_t> TalonInputFile::ClampReadSize(int64_t position, int64_t nbytes) const {
  if (position < 0) {
    return arrow::Status::Invalid("Cannot read from negative position");
  }
  if (nbytes < 0) {
    return arrow::Status::Invalid("Cannot read a negative number of bytes");
  }
  const int64_t size = identity_->content_length.load(std::memory_order_acquire);
  if (size != kNoSize) {
    if (position > size) {
      return arrow::Status::IOError("Cannot read past end of Talon object '", uri_, "'");
    }
    nbytes = std::min(nbytes, size - position);
  }
  return nbytes;
}

arrow::Future<int64_t> TalonInputFile::ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) {
  if (auto status = CheckClosed(); !status.ok()) {
    return FailedReadFuture(std::move(status));
  }
  auto clamped = ClampReadSize(position, nbytes);
  if (!clamped.ok()) {
    return FailedReadFuture(clamped.status());
  }
  nbytes = clamped.ValueOrDie();
  if (nbytes == 0) {
    return arrow::Future<int64_t>::MakeFinished(0);
  }

  // Take the stat-skipping fast path only when both halves of the identity are
  // known; otherwise let Talon resolve them and learn the result below.
  std::optional<std::string> version;
  std::optional<int64_t> object_size;
  const int64_t size = identity_->content_length.load(std::memory_order_acquire);
  if (size != kNoSize) {
    std::lock_guard<std::mutex> lock(identity_->version_mutex);
    if (identity_->has_version) {
      version = identity_->version;
      object_size = size;
    }
  }

  auto identity = identity_;
  return client_->ReadAsync(uri_, position, out, nbytes, version, object_size)
      .Then([identity, nbytes, uri = uri_](const TalonReadResult& result) -> arrow::Result<int64_t> {
        // Cache identity learned from a non-fast-path read so later reads skip
        // the stat. compare_exchange keeps the first resolved value.
        if (result.object_size != kNoSize) {
          int64_t expected = kNoSize;
          identity->content_length.compare_exchange_strong(expected, result.object_size, std::memory_order_acq_rel,
                                                           std::memory_order_acquire);
        }
        if (!result.version.empty()) {
          std::lock_guard<std::mutex> lock(identity->version_mutex);
          if (!identity->has_version) {
            identity->version = result.version;
            identity->has_version = true;
          }
        }
        if (result.bytes_written > nbytes) {
          return arrow::Status::IOError("Talon read for '", uri, "' returned ", result.bytes_written, " bytes for a ",
                                        nbytes, "-byte request");
        }
        return result.bytes_written;
      });
}

arrow::Result<int64_t> TalonInputFile::ReadAt(int64_t position, int64_t nbytes, void* out) {
  return ReadAtAsyncInto(position, nbytes, reinterpret_cast<uint8_t*>(out)).result();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> TalonInputFile::ReadAt(int64_t position, int64_t nbytes) {
  ARROW_RETURN_NOT_OK(CheckClosed());
  ARROW_ASSIGN_OR_RAISE(nbytes, ClampReadSize(position, nbytes));

  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes, pool_));
  if (nbytes > 0) {
    ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(position, nbytes, buffer->mutable_data()));
    ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
  }
  return std::shared_ptr<arrow::Buffer>(std::move(buffer));
}

arrow::Future<std::shared_ptr<arrow::Buffer>> TalonInputFile::ReadAsync(const arrow::io::IOContext& io_context,
                                                                        int64_t position,
                                                                        int64_t nbytes) {
  if (auto status = CheckClosed(); !status.ok()) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(std::move(status));
  }
  auto clamped = ClampReadSize(position, nbytes);
  if (!clamped.ok()) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(clamped.status());
  }
  nbytes = clamped.ValueOrDie();

  auto maybe_buffer = arrow::AllocateResizableBuffer(nbytes, io_context.pool());
  if (!maybe_buffer.ok()) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(maybe_buffer.status());
  }
  auto buffer = std::move(maybe_buffer).ValueOrDie();
  auto* out = buffer->mutable_data();

  return ReadAtAsyncInto(position, nbytes, out)
      .Then([buffer = std::move(buffer)](int64_t bytes_read) mutable -> arrow::Result<std::shared_ptr<arrow::Buffer>> {
        ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read));
        return std::shared_ptr<arrow::Buffer>(std::move(buffer));
      });
}

arrow::Result<int64_t> TalonInputFile::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(pos_, nbytes, out));
  pos_ += bytes_read;
  return bytes_read;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> TalonInputFile::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_ += buffer->size();
  return buffer;
}

arrow::Result<int64_t> TalonInputFile::GetSize() {
  ARROW_RETURN_NOT_OK(CheckClosed());
  ARROW_RETURN_NOT_OK(EnsureSize());
  return identity_->content_length.load(std::memory_order_acquire);
}

arrow::Status TalonInputFile::Seek(int64_t position) {
  ARROW_RETURN_NOT_OK(CheckClosed());
  if (position < 0) {
    return arrow::Status::Invalid("Cannot seek to a negative position");
  }
  const int64_t size = identity_->content_length.load(std::memory_order_acquire);
  if (size != kNoSize && position > size) {
    return arrow::Status::IOError("Cannot seek past end of Talon object '", uri_, "'");
  }
  pos_ = position;
  return arrow::Status::OK();
}

arrow::Result<int64_t> TalonInputFile::Tell() const {
  ARROW_RETURN_NOT_OK(CheckClosed());
  return pos_;
}

arrow::Status TalonInputFile::Close() {
  closed_ = true;
  client_.reset();
  return arrow::Status::OK();
}

bool TalonInputFile::closed() const { return closed_; }

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalonInputFile(
    std::shared_ptr<TalonClient> client, std::string uri, int64_t size, std::string version, arrow::MemoryPool* pool) {
  if (client == nullptr) {
    return arrow::Status::Invalid("OpenTalonInputFile requires a non-null Talon client");
  }
  return std::static_pointer_cast<arrow::io::RandomAccessFile>(
      std::make_shared<TalonInputFile>(std::move(client), std::move(uri), size, std::move(version), pool));
}

}  // namespace milvus_storage::talon
