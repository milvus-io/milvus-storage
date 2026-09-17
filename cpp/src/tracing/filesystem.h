// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arrow/io/interfaces.h>
#include <memory>
#include <string>
namespace milvus_storage::tracing {
// The wrapper stores the backend name only, never a request context or path.
std::shared_ptr<arrow::io::RandomAccessFile> WrapFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                                      std::string backend);
}  // namespace milvus_storage::tracing
