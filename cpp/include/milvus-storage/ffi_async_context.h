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

#pragma once

#include <stdint.h>
#include "milvus-storage/ffi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Caller-owned executor bridge. submit must be thread-safe and nonblocking.
 * Return 0 after accepting task(task_data) exactly once on your executor, never
 * inline; nonzero rejects it and MUST NOT invoke or retain task/task_data.
 * Keep submit and context alive and accepting work until loon_async_context_shutdown
 * returns for every async context using it, then drain/join your executor before freeing context. Storage never
 * creates or stops your pool. This executor runs both operation work and callbacks.
 */
typedef void (*LoonAsyncTask)(void* task_data);
typedef int32_t (*LoonAsyncSubmit)(void* context, LoonAsyncTask task, void* task_data);
typedef struct LoonAsyncExecutor {
  void* context;
  LoonAsyncSubmit submit;
} LoonAsyncExecutor;
/** Caller-owned scheduling context. Each context has independent admission and
 * shutdown state. Creation copies the executor descriptor, not its context.
 * The same executor may back multiple async contexts. No global configuration is required.
 */
typedef struct LoonAsyncContext* LoonAsyncContextHandle;
FFI_EXPORT LoonFFIResult loon_async_context_create(const LoonAsyncExecutor* executor,
                                                   LoonAsyncContextHandle* out_context);
/** Stop admission on this context and wait for its accepted callbacks to return.
 * Thread-safe and idempotent; may run concurrently with submissions. Call from
 * an application thread, never from a callback using this context.
 * Other contexts are unaffected. shutdown(NULL) is a no-op.
 */
FFI_EXPORT void loon_async_context_shutdown(LoonAsyncContextHandle async_context);
/** Shutdown and free the context. Exclude concurrent API calls using this context,
 * including submissions and shutdown; never call from one of its callbacks.
 * Operation handles remain valid for cancel/release after context destruction.
 * Does not stop the caller executor. destroy(NULL) is a no-op.
 */
FFI_EXPORT void loon_async_context_destroy(LoonAsyncContextHandle async_context);
/** Caller-owned operation handle. Zero-initialize before first use.
 * timeout_ms is copied at submission: zero selects 30 seconds; maximum is one day.
 * Deadlines are checked before execution and cannot interrupt synchronous I/O.
 * internal is library-owned; do not modify or copy a live handle. Release it before
 * reusing the handle. A failed submission leaves an empty handle empty.
 * Serialize submission, cancel and release calls on the same handle. Once
 * submission returns, the library never accesses the caller's handle storage.
 */
typedef struct LoonAsyncHandle {
  uint64_t timeout_ms;
  struct LoonAsyncOperation* internal;
} LoonAsyncHandle;
/** Request cancellation before execution. NULL/empty handles are no-ops. */
FFI_EXPORT void loon_async_cancel(LoonAsyncHandle* operation);
/** Drop and clear internal state without cancelling or waiting. The caller owns
 * the handle storage. NULL/empty handles are no-ops; timeout_ms is preserved.
 */
FFI_EXPORT void loon_async_release(LoonAsyncHandle* operation);

#ifdef __cplusplus
}
#endif
