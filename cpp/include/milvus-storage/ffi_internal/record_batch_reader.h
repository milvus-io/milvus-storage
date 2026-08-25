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

#pragma once

#include "milvus-storage/ffi_c.h"

#include <arrow/record_batch.h>

#include <memory>

namespace milvus_storage::ffi_internal {

struct RecordBatchReaderHolder {
  std::shared_ptr<arrow::RecordBatchReader> reader;
  /// Set when the stream ended normally. The underlying reader is released at
  /// that point, but further ReadNext calls keep reporting EOF (success with
  /// `release == nullptr`), matching the Arrow C stream convention. Only a
  /// reader made terminal by a FAILED read rejects further calls.
  bool exhausted = false;
};

/// Read one batch for a C/FFI consumer.
///
/// `materialize_offsets` is only needed by the JNI bridge, whose Arrow Java
/// importer does not honour ArrowArray.offset. Public C FFI callers should
/// leave it false and receive the standard zero-copy Arrow C Data export.
LoonFFIResult RecordBatchReaderReadNext(LoonRecordBatchReaderHandle handle,
                                        struct ArrowArray* out_array,
                                        bool materialize_offsets);

/// JNI-only offset materialization wrapper around the public read-next API.
LoonFFIResult RecordBatchReaderReadNextForJava(LoonRecordBatchReaderHandle handle,
                                               struct ArrowArray* out_array,
                                               struct ArrowSchema* out_schema);

}  // namespace milvus_storage::ffi_internal
