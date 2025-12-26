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

#include "milvus-storage/filesystem/s3/s3_delegator_filesystem.h"

#include <arrow/util/async_generator.h>
#include <arrow/util/logging.h>
#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/io/memory.h>
#include <arrow/util/future.h>
#include <arrow/util/thread_pool.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/io/interfaces.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/string.h>

#include "milvus-storage/filesystem/s3/s3_filesystem.h"

namespace milvus_storage {

S3DelegatorFileSystem::S3DelegatorFileSystem(std::shared_ptr<arrow::fs::FileSystem> base_fs) : base_fs_(base_fs) {}

arrow::Result<std::string> S3DelegatorFileSystem::NormalizePath(std::string path) {
  return base_fs_->NormalizePath(path);
}

arrow::Result<std::string> S3DelegatorFileSystem::PathFromUri(const std::string& uri_string) const {
  return base_fs_->PathFromUri(uri_string);
}

bool S3DelegatorFileSystem::Equals(const arrow::fs::FileSystem& other) const {
  if (this == &other) {
    return true;
  }
  if (other.type_name() != type_name()) {
    return false;
  }
  const auto& subfs = ::arrow::internal::checked_cast<const S3DelegatorFileSystem&>(other);
  return base_fs_->Equals(subfs.base_fs_);
}

arrow::Result<arrow::fs::FileInfo> S3DelegatorFileSystem::GetFileInfo(const std::string& path) {
  return base_fs_->GetFileInfo(path);
}

arrow::Result<arrow::fs::FileInfoVector> S3DelegatorFileSystem::GetFileInfo(const arrow::fs::FileSelector& select) {
  return base_fs_->GetFileInfo(select);
}

arrow::fs::FileInfoGenerator S3DelegatorFileSystem::GetFileInfoGenerator(const arrow::fs::FileSelector& select) {
  return base_fs_->GetFileInfoGenerator(select);
}

arrow::Status S3DelegatorFileSystem::CreateDir(const std::string& path, bool recursive) {
  return base_fs_->CreateDir(path, recursive);
}

arrow::Status S3DelegatorFileSystem::DeleteDir(const std::string& path) { return base_fs_->DeleteDir(path); }

arrow::Status S3DelegatorFileSystem::DeleteDirContents(const std::string& path, bool missing_dir_ok) {
  return base_fs_->DeleteDirContents(path, missing_dir_ok);
}

arrow::Status S3DelegatorFileSystem::DeleteRootDirContents() { return base_fs_->DeleteRootDirContents(); }

arrow::Status S3DelegatorFileSystem::DeleteFile(const std::string& path) { return base_fs_->DeleteFile(path); }

arrow::Status S3DelegatorFileSystem::Move(const std::string& src, const std::string& dest) {
  return base_fs_->Move(src, dest);
}

arrow::Status S3DelegatorFileSystem::CopyFile(const std::string& src, const std::string& dest) {
  return base_fs_->CopyFile(src, dest);
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> S3DelegatorFileSystem::OpenInputStream(const std::string& path) {
  return base_fs_->OpenInputStream(path);
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> S3DelegatorFileSystem::OpenInputStream(
    const arrow::fs::FileInfo& info) {
  return base_fs_->OpenInputStream(info);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> S3DelegatorFileSystem::OpenInputFile(
    const std::string& path) {
  return base_fs_->OpenInputFile(path);
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> S3DelegatorFileSystem::OpenInputFile(
    const arrow::fs::FileInfo& info) {
  return base_fs_->OpenInputFile(info);
}

arrow::Future<std::shared_ptr<arrow::io::InputStream>> S3DelegatorFileSystem::OpenInputStreamAsync(
    const std::string& path) {
  return base_fs_->OpenInputStreamAsync(path);
}

arrow::Future<std::shared_ptr<arrow::io::InputStream>> S3DelegatorFileSystem::OpenInputStreamAsync(
    const arrow::fs::FileInfo& info) {
  return base_fs_->OpenInputStreamAsync(info);
}

arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile>> S3DelegatorFileSystem::OpenInputFileAsync(
    const std::string& path) {
  return base_fs_->OpenInputFileAsync(path);
}
arrow::Future<std::shared_ptr<arrow::io::RandomAccessFile>> S3DelegatorFileSystem::OpenInputFileAsync(
    const arrow::fs::FileInfo& info) {
  return base_fs_->OpenInputFileAsync(info);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3DelegatorFileSystem::OpenOutputStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  std::shared_ptr<S3FileSystem> base_s3_fs = std::dynamic_pointer_cast<S3FileSystem>(base_fs_);

  if (metadata && base_s3_fs) {
    int64_t part_upload_size = -1;
    int64_t keyidx = metadata->FindKey(kMultiPartUploadSizeKey);
    if (keyidx != -1) {
      auto part_size_str = metadata->value(keyidx);

      int64_t part_size_result;
      auto [ptr, ec] =
          std::from_chars(part_size_str.data(), part_size_str.data() + part_size_str.size(), part_size_result);

      if (ec == std::errc() && ptr == part_size_str.data() + part_size_str.size()) {
        part_upload_size = part_size_result;
      } else {
        ARROW_LOG(WARNING) << "Failed to parse MultiPartUploadSize: " << part_size_str;
      }
    }

    if (part_upload_size != -1) {
      if (part_upload_size < MINIMAL_MULTIPART_UPLOAD_PART_SIZE ||
          part_upload_size > MAXIMAL_MULTIPART_UPLOAD_PART_SIZE) {
        return arrow::Status::Invalid("Invalid MultiPartUploadSize: ", part_upload_size);
      }

      // re-construct the metadata
      auto new_metadata = metadata->Copy();
      ARROW_RETURN_NOT_OK(new_metadata->Delete(keyidx));
      return base_s3_fs->OpenOutputStreamWithUploadSize(path, new_metadata, part_upload_size);
    }
  }

  return base_fs_->OpenOutputStream(path, metadata);
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> S3DelegatorFileSystem::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
  return base_fs_->OpenAppendStream(path, metadata);
}

}  // namespace milvus_storage
