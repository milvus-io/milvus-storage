// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <arrow/buffer.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage {

/// Missing filesystem-level asynchronous operations with native network requests. There is no
/// thread-pool fallback to synchronous FileSystem methods. Paths have the same
/// meaning as on the FileSystem passed to MakeAsyncFileSystem (including subtrees).
///
/// Creation initializes transport/configuration and is not an asynchronous
/// operation. Keep the supplied executor alive and running until all operations
/// finish. Future continuations normally run there; executor rejection completes
/// inline with the original I/O result. Do not block inside an inline continuation.
class AsyncFileSystem {
  public:
  virtual ~AsyncFileSystem() = default;
  virtual arrow::Future<arrow::fs::FileInfo> GetFileInfoAsync(const std::string& path) = 0;
  /// Pages are fetched on demand; only one outstanding invocation per generator.
  virtual arrow::fs::FileInfoGenerator GetFileInfoGenerator(const arrow::fs::FileSelector& selector) = 0;

};

/// Returns NotImplemented for providers/configurations without a native async
/// transport, including builds without WITH_CRT. Existing sync APIs are unchanged.
arrow::Result<std::shared_ptr<AsyncFileSystem>> MakeAsyncFileSystem(std::shared_ptr<arrow::fs::FileSystem> filesystem,
                                                                    const arrow::io::IOContext& io_context);

}  // namespace milvus_storage
