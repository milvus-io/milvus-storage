// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifdef WITH_CRT
#include <arrow/filesystem/filesystem.h>
#include "filesystem/s3/native_s3_transport.h"

namespace milvus_storage {
struct NativeS3ObjectMetadata {
  int64_t content_length;
  std::shared_ptr<const arrow::KeyValueMetadata> metadata;
};

// Private implementation owned by S3FileSystem, never handed to callers.
class NativeS3Operations {
  public:
  virtual ~NativeS3Operations() = default;
  virtual arrow::Future<NativeS3ObjectMetadata> ReadMetadataAsync(const std::string& path,
                                                                  const arrow::io::IOContext& io_context) = 0;
  virtual arrow::Future<arrow::fs::FileInfo> GetFileInfoAsync(const std::string& path) = 0;
  virtual arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) = 0;
  virtual arrow::Future<> CreateDirAsync(const std::string& path, bool recursive) = 0;
  virtual arrow::Future<> DeleteDirAsync(const std::string& path) = 0;
  virtual arrow::Future<> DeleteFileAsync(const std::string& path) = 0;
  virtual arrow::Future<> CopyFileAsync(const std::string& source, const std::string& destination) = 0;
  virtual arrow::Future<> MoveAsync(const std::string& source, const std::string& destination) = 0;
  virtual arrow::Future<> DeleteDirContentsAsync(const std::string& path, bool missing_dir_ok) = 0;
  virtual arrow::Future<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStreamAsync(
      const std::string& path, const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) = 0;
};
arrow::Result<std::shared_ptr<NativeS3Operations>> MakeNativeS3Operations(const S3Options& options,
                                                                          const arrow::io::IOContext& io_context,
                                                                          std::shared_ptr<NativeS3Transport> transport);
arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenNativeS3OutputStream(
    const S3Options& options,
    std::shared_ptr<NativeS3Transport> transport,
    const arrow::io::IOContext& io_context,
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata);
}  // namespace milvus_storage
#endif
