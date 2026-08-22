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
#include <utility>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>

#include "milvus-storage/common/writer_status.h"

template <template <typename> class P, typename T>
class PolymorphicWrapper {
  public:
  PolymorphicWrapper(P<T> obj) : obj_(obj) {}
  inline P<T> get() const { return obj_; }

  private:
  P<T> obj_;
};

using FileSystemWrapper = PolymorphicWrapper<std::shared_ptr, arrow::fs::FileSystem>;
using RandomAccessFileWrapper = PolymorphicWrapper<std::shared_ptr, arrow::io::RandomAccessFile>;

class OutputStreamWrapper {
  public:
  explicit OutputStreamWrapper(std::shared_ptr<arrow::io::OutputStream> stream) : stream_(std::move(stream)) {}

  ~OutputStreamWrapper() = default;

  arrow::Status Write(const void* data, int64_t nbytes) {
    ARROW_RETURN_NOT_OK(writer_status_.Check());
    return writer_status_.Fail(stream_->Write(data, nbytes));
  }

  arrow::Status Flush() {
    ARROW_RETURN_NOT_OK(writer_status_.Check());
    return writer_status_.Fail(stream_->Flush());
  }

  arrow::Status Close() {
    ARROW_RETURN_NOT_OK(writer_status_.Check());
    return writer_status_.Fail(stream_->Close());
  }

  /// Give up on the stream and release what it allocated in the store.
  ///
  /// Deliberately no writer_status_.Check(): this is the one verb that has to
  /// work after the writer has failed. Best effort -- the stream's own cleanup
  /// failure is not reported, because whoever calls this is already handling a
  /// failure of their own.
  ///
  /// The Rust bridge reaches this through loon_filesystem_writer_destroy, which
  /// it already calls on every terminal path -- failed write, failed flush,
  /// close, and Drop of an open writer (filesystem_c.rs). So a vortex file
  /// abandoned mid-write does get its upload cancelled; the release rides the C
  /// ABI the bridge already uses rather than a new cxx verb.
  ///
  /// Lance drives its own Rust object_store and never this wrapper, so nothing
  /// here can reach its uncommitted fragments -- which costs nothing, because
  /// the lance writer only exists in test builds.
  arrow::Status Abort() {
    writer_status_.BeginDiscard();
    return stream_->Abort();
  }

  private:
  std::shared_ptr<arrow::io::OutputStream> stream_;
  milvus_storage::WriterStatus writer_status_;
};
