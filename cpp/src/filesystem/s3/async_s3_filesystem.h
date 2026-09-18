// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifdef WITH_CRT
#include <arrow/filesystem/filesystem.h>
#include "filesystem/s3/native_s3_transport.h"

namespace milvus_storage {
// Private implementation owned by S3FileSystem, never handed to callers.
class NativeS3Operations {
  public:
  virtual ~NativeS3Operations() = default;
  virtual arrow::Future<arrow::fs::FileInfo> GetFileInfoAsync(const std::string& path) = 0;
  virtual arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) = 0;
};
arrow::Result<std::shared_ptr<NativeS3Operations>> MakeNativeS3Operations(const S3Options& options,
                                                                          const arrow::io::IOContext& io_context,
                                                                          std::shared_ptr<NativeS3Transport> transport);
}  // namespace milvus_storage
#endif
