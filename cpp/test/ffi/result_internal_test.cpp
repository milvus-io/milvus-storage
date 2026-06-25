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

#include <gtest/gtest.h>

#include <arrow/io/memory.h>
#include <arrow/status.h>

#include "milvus-storage/common/extend_status.h"
#include "milvus-storage/ffi_filesystem_c.h"
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/filesystem/async_random_access_file.h"
#include "milvus-storage/filesystem/ffi/filesystem_internal.h"

namespace milvus_storage::test {
namespace {

LoonFFIResult ReturnArrowErrorIf(const arrow::Status& status, int fallback) {
  RETURN_ARROW_ERROR_IF(status, fallback, status.ToString());
  RETURN_SUCCESS();
}

LoonFFIResult ReturnArrowError(const arrow::Status& status, int fallback) {
  RETURN_ARROW_ERROR(status, fallback, status.ToString());
}

class FailingAsyncRandomAccessFile final : public arrow::io::RandomAccessFile, public NonBlockingReadAtFile {
  public:
  explicit FailingAsyncRandomAccessFile(arrow::Status status)
      : file_(std::make_shared<arrow::io::BufferReader>("test payload")), status_(std::move(status)) {}

  arrow::Status Close() override { return file_->Close(); }
  arrow::Status Abort() override { return file_->Abort(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override { return file_->Read(nbytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override { return file_->Read(nbytes); }
  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }
  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    return file_->ReadMetadata();
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& io_context) override {
    return file_->ReadMetadataAsync(io_context);
  }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }
  arrow::Result<int64_t> GetSize() override { return file_->GetSize(); }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    return file_->ReadAt(position, nbytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    return file_->ReadAt(position, nbytes);
  }
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t, int64_t, uint8_t*) override {
    return arrow::Future<int64_t>::MakeFinished(status_);
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& io_context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    return file_->ReadAsync(io_context, position, nbytes);
  }
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }

  private:
  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  arrow::Status status_;
};

struct AsyncReadCallbackCapture {
  int callback_count = 0;
  LoonFFIResult result{LOON_SUCCESS, nullptr};
  uint64_t bytes_read = 0;
};

void CaptureAsyncReadResult(void* user_data, LoonFFIResult result, uint64_t bytes_read) {
  auto* capture = static_cast<AsyncReadCallbackCapture*>(user_data);
  capture->callback_count++;
  capture->result = result;
  capture->bytes_read = bytes_read;
}

}  // namespace

TEST(FFIInternalResultTest, MapsStatusDetailsToFfiResults) {
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(arrow::Status::Invalid("plain"), LOON_LOGICAL_ERROR), LOON_LOGICAL_ERROR);

  auto timeout_status = MakeExtendError(ExtendStatusCode::StorageTransientTimeout, "timeout", "timeout");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(timeout_status, LOON_ARROW_ERROR), LOON_TRANSIENT_TIMEOUT);

  auto code = ExtendStatusCodeFromFFIErrorCode(LOON_AWS_ERROR_NO_SUCH_UPLOAD);
  ASSERT_TRUE(code.has_value());
  EXPECT_EQ(*code, ExtendStatusCode::AwsErrorNoSuchUpload);

  code = ExtendStatusCodeFromFFIErrorCode(LOON_TRANSIENT_TIMEOUT);
  ASSERT_TRUE(code.has_value());
  EXPECT_EQ(*code, ExtendStatusCode::StorageTransientTimeout);
  EXPECT_FALSE(ExtendStatusCodeFromFFIErrorCode(LOON_ARROW_ERROR).has_value());

  auto conflict_status = MakeExtendError(ExtendStatusCode::AwsErrorConflict, "conflict", "conflict");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(conflict_status, LOON_ARROW_ERROR), LOON_AWS_ERROR_CONFLICT);
  code = ExtendStatusCodeFromFFIErrorCode(LOON_AWS_ERROR_CONFLICT);
  ASSERT_TRUE(code.has_value());
  EXPECT_EQ(*code, ExtendStatusCode::AwsErrorConflict);

  auto precondition_status =
      MakeExtendError(ExtendStatusCode::AwsErrorPreConditionFailed, "precondition", "precondition");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(precondition_status, LOON_ARROW_ERROR), LOON_AWS_ERROR_PRECONDITION_FAILED);
  code = ExtendStatusCodeFromFFIErrorCode(LOON_AWS_ERROR_PRECONDITION_FAILED);
  ASSERT_TRUE(code.has_value());
  EXPECT_EQ(*code, ExtendStatusCode::AwsErrorPreConditionFailed);

  auto not_found_status = MakeExtendError(ExtendStatusCode::AwsErrorNotFound, "missing", "missing");
  EXPECT_EQ(FFIErrorCodeFromExtendStatus(not_found_status, LOON_ARROW_ERROR), LOON_AWS_ERROR_NOT_FOUND);
  code = ExtendStatusCodeFromFFIErrorCode(LOON_AWS_ERROR_NOT_FOUND);
  ASSERT_TRUE(code.has_value());
  EXPECT_EQ(*code, ExtendStatusCode::AwsErrorNotFound);

  EXPECT_TRUE(loon_ffi_is_retryable_errcode(LOON_AWS_ERROR_NO_SUCH_UPLOAD));
  EXPECT_FALSE(loon_ffi_is_retryable_errcode(LOON_AWS_ERROR_ACCESS_DENIED));

  auto throttling_status = MakeExtendError(ExtendStatusCode::StorageTransientThrottling, "throttled", "throttled");
  auto throttling_result = ReturnArrowErrorIf(throttling_status, LOON_ARROW_ERROR);
  EXPECT_EQ(throttling_result.err_code, LOON_TRANSIENT_THROTTLING);
  ASSERT_NE(throttling_result.message, nullptr);
  EXPECT_NE(std::string(throttling_result.message).find("throttled"), std::string::npos);
  loon_ffi_free_result(&throttling_result);

  auto network_status = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network", "network");
  auto network_result = ReturnArrowError(network_status, LOON_ARROW_ERROR);
  EXPECT_EQ(network_result.err_code, LOON_TRANSIENT_NETWORK);
  ASSERT_NE(network_result.message, nullptr);
  EXPECT_NE(std::string(network_result.message).find("network"), std::string::npos);
  loon_ffi_free_result(&network_result);

  auto ok_result = ReturnArrowErrorIf(arrow::Status::OK(), LOON_ARROW_ERROR);
  EXPECT_EQ(ok_result.err_code, LOON_SUCCESS);
  EXPECT_EQ(ok_result.message, nullptr);
}

TEST(FFIInternalResultTest, AsyncReadCallbackPreservesExtendStatusCode) {
  auto status = MakeExtendError(ExtendStatusCode::StorageTransientNetwork, "network", "network detail");
  auto file = std::make_shared<FailingAsyncRandomAccessFile>(status);
  auto wrapper = std::make_unique<RandomAccessFileWrapper>(std::move(file));
  auto handle = reinterpret_cast<FileSystemReaderHandle>(wrapper.get());
  uint8_t buffer[4] = {};
  AsyncReadCallbackCapture capture;

  auto submit_result =
      loon_filesystem_reader_readat_async(handle, 0, sizeof(buffer), buffer, CaptureAsyncReadResult, &capture);

  EXPECT_EQ(submit_result.err_code, LOON_SUCCESS);
  EXPECT_EQ(submit_result.message, nullptr);
  EXPECT_EQ(capture.callback_count, 1);
  EXPECT_EQ(capture.result.err_code, LOON_TRANSIENT_NETWORK);
  EXPECT_EQ(capture.bytes_read, 0);
  ASSERT_NE(capture.result.message, nullptr);
  EXPECT_NE(std::string(capture.result.message).find("network"), std::string::npos);
  loon_ffi_free_result(&capture.result);
}

}  // namespace milvus_storage::test
