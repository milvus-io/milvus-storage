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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/result.h>
#include <arrow/util/future.h>

#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/talon/talon_client.h"

namespace milvus_storage::talon {

/// A random-access input file backed by the Talon C SDK.
///
/// It implements arrow::io::RandomAccessFile, which is the single interface
/// every Arrow-based format reader consumes (Parquet, Lance, Arrow IPC, CSV,
/// ...). Constructing one over a Talon URI therefore lets any of those formats
/// be read straight out of the Talon cache with no per-format work. It also
/// implements milvus_storage::NonBlockingReadAtFile, the async fast path the
/// Parquet reader opportunistically `dynamic_cast`s to, so range reads issue
/// through `talon_read_async` and complete without blocking a worker thread.
///
/// Object identity (length + version) is resolved lazily and cached: the first
/// GetSize() or the first read learns both from Talon, and every read after
/// that supplies them so Talon skips its StatObject round trip (the PR #564
/// fast path). The buffer handed to a read must outlive the returned future —
/// that is the NonBlockingReadAtFile contract, and Talon owns the buffer until
/// completion.
class TalonInputFile final : public arrow::io::RandomAccessFile, public NonBlockingReadAtFile {
  public:
  /// @param client Shared Talon client; kept alive for the file's lifetime.
  /// @param uri    Object URI Talon resolves (e.g. "s3://bucket/key").
  /// @param size   Known object length, or arrow::fs::kNoSize if unknown.
  /// @param version Known object version, or "" if unknown. A size+version pair
  ///        known up front lets the very first read take the fast path.
  /// @param pool   Memory pool for buffer-returning reads.
  TalonInputFile(std::shared_ptr<TalonClient> client,
                 std::string uri,
                 int64_t size = -1,
                 std::string version = "",
                 arrow::MemoryPool* pool = arrow::default_memory_pool());

  ~TalonInputFile() override = default;

  // arrow::io::RandomAccessFile
  arrow::Result<int64_t> GetSize() override;
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override;
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override;

  // arrow::io::Seekable / InputStream
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;
  arrow::Status Seek(int64_t position) override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Status Close() override;
  bool closed() const override;

  // milvus_storage::NonBlockingReadAtFile
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override;

  private:
  /// Shared read identity, updated at most once from unknown -> known. The
  /// content length is atomic (the hot read path only reads it); the version is
  /// guarded because it is a string and coupled with `has_version`.
  struct Identity {
    explicit Identity(int64_t size) : content_length(size) {}
    std::atomic<int64_t> content_length;  // arrow::fs::kNoSize until resolved
    mutable std::mutex version_mutex;
    std::string version;
    bool has_version = false;
  };

  arrow::Status CheckClosed() const;
  /// Resolve and cache the object length (and version) if not already known.
  arrow::Status EnsureSize();
  /// Clamp a requested read to [0, size - position] when the size is known;
  /// pass it through unchanged (letting Talon bound it at EOF) when it is not.
  arrow::Result<int64_t> ClampReadSize(int64_t position, int64_t nbytes) const;

  std::shared_ptr<TalonClient> client_;
  std::string uri_;
  std::shared_ptr<Identity> identity_;
  arrow::MemoryPool* pool_;
  int64_t pos_ = 0;
  bool closed_ = false;
};

/// Open a Talon-backed random-access file. The returned handle plugs directly
/// into any Arrow format reader that takes an arrow::io::RandomAccessFile.
arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenTalonInputFile(
    std::shared_ptr<TalonClient> client,
    std::string uri,
    int64_t size = -1,
    std::string version = "",
    arrow::MemoryPool* pool = arrow::default_memory_pool());

}  // namespace milvus_storage::talon
