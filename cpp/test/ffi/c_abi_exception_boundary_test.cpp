// Copyright 2026 Zilliz
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

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/c/abi.h>

#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_filesystem_c.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"
#include "milvus-storage/writer.h"

namespace milvus_storage::test {
namespace {

class NonStdThrowingWriter final : public api::Writer {
  public:
  NonStdThrowingWriter(bool* abort_called, bool* destroyed) : abort_called_(abort_called), destroyed_(destroyed) {}

  ~NonStdThrowingWriter() override { *destroyed_ = true; }

  std::shared_ptr<arrow::Schema> schema() const override { return arrow::schema({}); }

  arrow::Status write(const std::shared_ptr<arrow::RecordBatch>&) override {
    return arrow::Status::NotImplemented("unused");
  }

  arrow::Status flush() override { throw 42; }

  arrow::Result<std::shared_ptr<api::ColumnGroups>> close(const std::vector<std::string_view>&,
                                                          const std::vector<std::string_view>&) override {
    return arrow::Status::NotImplemented("unused");
  }

  // Not throwing: Writer::abort() is noexcept by contract, and a stub that
  // violated it would terminate rather than exercise anything. The non-std
  // exception this test is about comes from flush() above; what is asserted
  // here is that destroy still reaches abort and still deletes.
  void abort() noexcept override { *abort_called_ = true; }

  private:
  bool* abort_called_;
  bool* destroyed_;
};

class NonStdThrowingOutputStream final : public arrow::io::OutputStream {
  public:
  NonStdThrowingOutputStream(bool* abort_called, bool* destroyed)
      : abort_called_(abort_called), destroyed_(destroyed) {}

  ~NonStdThrowingOutputStream() override { *destroyed_ = true; }

  arrow::Status Close() override { return arrow::Status::OK(); }

  arrow::Status Abort() override {
    *abort_called_ = true;
    throw 42;
  }

  arrow::Result<int64_t> Tell() const override { return 0; }

  bool closed() const override { return false; }

  arrow::Status Write(const void*, int64_t) override { return arrow::Status::OK(); }

  arrow::Status Flush() override { return arrow::Status::OK(); }

  private:
  bool* abort_called_;
  bool* destroyed_;
};

void ThrowingArrowArrayRelease(ArrowArray* array) {
  auto* release_count = static_cast<int*>(array->private_data);
  ++*release_count;
  throw 42;
}

}  // namespace

TEST(CAbiExceptionBoundaryTest, ConvertsNonStdExceptionAndDestroyStillDeletes) {
  bool abort_called = false;
  bool destroyed = false;
  auto* writer = new NonStdThrowingWriter(&abort_called, &destroyed);
  auto handle = reinterpret_cast<LoonWriterHandle>(writer);

  auto result = loon_writer_flush(handle);
  EXPECT_EQ(result.err_code, LOON_GOT_EXCEPTION);
  loon_ffi_free_result(&result);

  EXPECT_NO_THROW(loon_writer_destroy(handle));
  EXPECT_TRUE(abort_called);
  EXPECT_TRUE(destroyed);
}

TEST(CAbiExceptionBoundaryTest, FilesystemWriterDestroyStillDeletesAfterAbortThrows) {
  bool abort_called = false;
  bool destroyed = false;
  auto stream = std::make_shared<NonStdThrowingOutputStream>(&abort_called, &destroyed);
  auto* wrapper = new OutputStreamWrapper(stream);
  stream.reset();

  EXPECT_NO_THROW(loon_filesystem_writer_destroy(reinterpret_cast<FileSystemWriterHandle>(wrapper)));
  EXPECT_TRUE(abort_called);
  EXPECT_TRUE(destroyed);
}

TEST(CAbiExceptionBoundaryTest, ArrowArrayCleanupContinuesAfterReleaseThrows) {
  int release_count = 0;
  auto* arrays = static_cast<ArrowArray*>(std::calloc(2, sizeof(ArrowArray)));
  ASSERT_NE(arrays, nullptr);
  for (size_t i = 0; i < 2; ++i) {
    arrays[i].release = ThrowingArrowArrayRelease;
    arrays[i].private_data = &release_count;
  }

  EXPECT_NO_THROW(loon_free_chunk_arrays(arrays, 2));
  EXPECT_EQ(release_count, 2);
}

}  // namespace milvus_storage::test
