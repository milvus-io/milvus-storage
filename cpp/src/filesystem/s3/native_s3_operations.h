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
class NativeS3Operations : public std::enable_shared_from_this<NativeS3Operations> {
  public:
  NativeS3Operations(std::shared_ptr<NativeS3Transport> transport, arrow::io::IOContext io);
  arrow::Future<NativeS3ObjectMetadata> ReadMetadataAsync(const std::string& path,
                                                          const arrow::io::IOContext& io_context);
  arrow::Future<arrow::fs::FileInfo> GetFileInfoAsync(const std::string& path);
  arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector);

  private:
  struct Path;
  arrow::Future<NativeS3Response> Head(const Path& path, bool marker = false);
  arrow::Future<NativeS3Response> List(const Path& path, const std::string& token, bool recursive, int max_keys = 1000);

  std::shared_ptr<NativeS3Transport> transport_;
  arrow::io::IOContext io_;
};
arrow::Result<std::shared_ptr<NativeS3Operations>> MakeNativeS3Operations(const arrow::io::IOContext& io_context,
                                                                          std::shared_ptr<NativeS3Transport> transport);
}  // namespace milvus_storage
#endif
