// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#ifdef WITH_CRT
#include "milvus-storage/filesystem/async_filesystem.h"
#include "filesystem/s3/native_s3_transport.h"

namespace milvus_storage {
arrow::Result<std::shared_ptr<AsyncFileSystem>> MakeAsyncS3FileSystem(const S3Options& options,
                                                                      std::shared_ptr<S3ClientHolder> holder,
                                                                      const arrow::io::IOContext& io_context);
}  // namespace milvus_storage
#endif
