// Copyright 2023 Zilliz
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

#include "milvus-storage/ffi_internal/async_context.h"
#include "milvus-storage/ffi_internal/result.h"

using namespace milvus_storage;

LoonFFIResult loon_async_context_create(const LoonAsyncExecutor* executor, LoonAsyncContextHandle* out_context) {
  if (out_context)
    *out_context = nullptr;
  if (!out_context || !executor || !executor->submit)
    return {LOON_INVALID_ARGS, nullptr};
  try {
    *out_context = new LoonAsyncContext(*executor);
    return {LOON_SUCCESS, nullptr};
  } catch (const std::bad_alloc&) {
    return {LOON_MEMORY_ERROR, nullptr};
  } catch (...) {
    return {LOON_GOT_EXCEPTION, nullptr};
  }
}

void loon_async_cancel(LoonAsyncHandle* operation) {
  if (operation && operation->internal)
    operation->internal->state->Cancel();
}
void loon_async_release(LoonAsyncHandle* operation) {
  if (operation) {
    delete operation->internal;
    operation->internal = nullptr;
  }
}
void loon_async_context_shutdown(LoonAsyncContextHandle async_context) {
  if (async_context)
    async_context->Shutdown();
}
void loon_async_context_destroy(LoonAsyncContextHandle async_context) { delete async_context; }

LoonFFIResult milvus_storage::ValidateAsyncHandle(const LoonAsyncHandle* operation, uint64_t& timeout) {
  if (!operation)
    return {LOON_INVALID_ARGS, nullptr};
  if (operation->internal)
    return {LOON_ASYNC_BUSY, nullptr};
  timeout = operation->timeout_ms ? operation->timeout_ms : 30000;
  if (timeout > 24 * 60 * 60 * 1000)
    return CreateFFIResult(LOON_INVALID_ARGS, "Timeout exceeds one day");
  return {LOON_SUCCESS, nullptr};
}
