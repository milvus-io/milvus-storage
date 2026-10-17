// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <cstdint>
#include <arrow/util/future.h>
namespace milvus_storage {
/// Only the operations missing from Arrow OutputStream. Data still uses Write;
/// publication still uses its existing CloseAsync. Calls on a stream are serialized
/// by the caller. Keep input buffers immutable until FlushAsync/CloseAsync finishes.
class AsyncOutputStream {
  public:
  virtual ~AsyncOutputStream() = default;
  virtual arrow::Future<> FlushAsync() = 0;
  /// Wait before retrying a capacity-rejected Write(nbytes), without flushing
  /// every pending request. A retry may still reject; a write larger than the
  /// entire request window fails immediately. Unsupported streams return NotImplemented.
  virtual arrow::Future<> WaitForCapacityAsync(int64_t nbytes) = 0;
  virtual arrow::Future<> AbortAsync() = 0;
};
}  // namespace milvus_storage
