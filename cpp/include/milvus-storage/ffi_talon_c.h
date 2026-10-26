// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
#pragma once

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_async_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Independent raw client configured with fs.talon.coordinator,
/// fs.talon.block_size, and fs.talon.max_idle_per_addr. Optional fs.talon.namespace
/// and fs.credential_endpoint are independent request defaults. fs.credential_endpoint
/// uses the fs.* cloud identity and origin settings for every Stat/Read/Load request.
/// Property-derived Broker defaults are scoped to fs.bucket_name. Per-request
/// Broker overrides supply their own descriptor; URI buckets must match it.
/// Broker-backed GCP objects use gcs:// URIs; S3 and Azure use s3:// and az://.
typedef uintptr_t LoonTalonClientHandle;

/// Optional overrides for one operation, without changing the shared client.
/// NULL options or NULL fields inherit fs.talon.namespace / fs.credential_endpoint
/// defaults captured at client creation. An empty string explicitly clears an
/// option: default namespace / worker-side static credentials, respectively.
/// Async calls copy both strings before returning; callers may then release them.
typedef struct LoonTalonRequestOptions {
  const char* namespace_name;
  /// Complete UTF-8 JSON {url, request, origin:{endpoint,...}}, at most 16 KiB.
  /// S3 descriptors also require origin.region. No cloud access keys are sent.
  const char* credential_broker;
} LoonTalonRequestOptions;
typedef enum LoonTalonOperation {
  /// Read object bytes into the caller's buffer.
  LOON_TALON_READ = 0,
  /// Fetch the object size and version.
  LOON_TALON_STAT = 1,
  /// Prewarm one object using its known size and optional version.
  LOON_TALON_LOAD = 2,
  /// Prewarm multiple objects in one native batch.
  LOON_TALON_BATCH_LOAD = 3
} LoonTalonOperation;

typedef struct LoonTalonLoadRequest {
  const char* uri;
  uint64_t size;        ///< Mandatory known size; zero is an empty file.
  const char* version;  ///< NULL or empty skips version matching.
} LoonTalonLoadRequest;

typedef struct LoonTalonLoadResult {
  uint64_t size;
  uint64_t blocks;
} LoonTalonLoadResult;

typedef struct LoonTalonLoadFailure {
  uint64_t index;  ///< Index in the original request array.
  bool uncertain;  ///< Upstream cannot confirm whether this file was loaded.
  char* error;     ///< Complete upstream diagnostic.
} LoonTalonLoadFailure;

/// Library-owned payload; operation selects the active data member.
/// Successful batch results follow input order. Failed batches contain only
/// failure details because per-file success details are unknown. Free even on failure.
typedef struct LoonTalonResult {
  LoonTalonOperation operation;
  union {
    /// READ: number of bytes written to the caller's buffer.
    int64_t bytes_read;
    /// STAT: object metadata.
    struct {
      uint64_t size;
      char* version;
    } stat;
    /// LOAD: one result. BATCH_LOAD: success results or failure details.
    struct {
      bool failed;  ///< Selects entries.failures when true, entries.results otherwise.
      size_t count;
      union {
        LoonTalonLoadResult* results;
        LoonTalonLoadFailure* failures;
      } entries;
    } load;
  } data;
} LoonTalonResult;
/// Callback owns both status and payload. Free status with loon_ffi_free_result
/// and payload with loon_talon_result_free.
typedef void (*LoonTalonCallback)(uintptr_t user_data, LoonFFIResult status, LoonTalonResult* result);

FFI_EXPORT LoonFFIResult loon_talon_client_create(const LoonProperties* properties, LoonTalonClientHandle* client);
/// Accepted operations retain the client. Exclude concurrent handle calls during destroy;
/// destroying a zero handle is a no-op.
FFI_EXPORT void loon_talon_client_destroy(LoonTalonClientHandle client);
/// Blocking calls initialize *result to NULL before validation. Partial batch
/// failures may return a payload. Raw operations work regardless of filesystem
/// routing mode or threshold and never fall back to the origin.
/// READ: NULL object_size means unknown; a present zero means known empty.
/// Nonempty version requires object_size. dst may be NULL only for length zero.
/// URI, size, and version must describe the same object, including prewarm
/// followed by normal reads.
FFI_EXPORT LoonFFIResult loon_talon_read(LoonTalonClientHandle client,
                                         const char* uri,
                                         uint64_t offset,
                                         uint64_t length,
                                         const char* version,
                                         const uint64_t* object_size,
                                         uint8_t* dst,
                                         const LoonTalonRequestOptions* options,
                                         LoonTalonResult** result);
FFI_EXPORT LoonFFIResult loon_talon_stat(LoonTalonClientHandle client,
                                         const char* uri,
                                         const LoonTalonRequestOptions* options,
                                         LoonTalonResult** result);
FFI_EXPORT LoonFFIResult loon_talon_load(LoonTalonClientHandle client,
                                         const LoonTalonLoadRequest* request,
                                         const LoonTalonRequestOptions* options,
                                         LoonTalonResult** result);
FFI_EXPORT LoonFFIResult loon_talon_batch_load(LoonTalonClientHandle client,
                                               const LoonTalonLoadRequest* requests,
                                               size_t count,
                                               const LoonTalonRequestOptions* options,
                                               LoonTalonResult** result);
/// Async inputs except dst are copied before successful admission. Rejection
/// returns an error synchronously, starts no I/O, and calls no callback.
/// Accepted work calls back exactly once on the context's caller executor,
/// subject to its existing queue-rejection fallback. Namespace and Broker
/// validation errors from the Talon SDK are reported through that callback.
/// Cancellation and deadline are checked only before native dispatch; native
/// timeouts govern in-flight I/O.
/// Keep dst valid until the actual callback, even after cancellation. Releasing
/// the operation handle does not cancel accepted work.
FFI_EXPORT LoonFFIResult loon_talon_read_async(LoonAsyncContextHandle context,
                                               LoonTalonClientHandle client,
                                               const char* uri,
                                               uint64_t offset,
                                               uint64_t length,
                                               const char* version,
                                               const uint64_t* object_size,
                                               uint8_t* dst,
                                               const LoonTalonRequestOptions* options,
                                               LoonTalonCallback callback,
                                               uintptr_t user_data,
                                               LoonAsyncHandle* operation);
FFI_EXPORT LoonFFIResult loon_talon_stat_async(LoonAsyncContextHandle context,
                                               LoonTalonClientHandle client,
                                               const char* uri,
                                               const LoonTalonRequestOptions* options,
                                               LoonTalonCallback callback,
                                               uintptr_t user_data,
                                               LoonAsyncHandle* operation);
FFI_EXPORT LoonFFIResult loon_talon_load_async(LoonAsyncContextHandle context,
                                               LoonTalonClientHandle client,
                                               const LoonTalonLoadRequest* request,
                                               const LoonTalonRequestOptions* options,
                                               LoonTalonCallback callback,
                                               uintptr_t user_data,
                                               LoonAsyncHandle* operation);
FFI_EXPORT LoonFFIResult loon_talon_batch_load_async(LoonAsyncContextHandle context,
                                                     LoonTalonClientHandle client,
                                                     const LoonTalonLoadRequest* requests,
                                                     size_t count,
                                                     const LoonTalonRequestOptions* options,
                                                     LoonTalonCallback callback,
                                                     uintptr_t user_data,
                                                     LoonAsyncHandle* operation);
/// Frees the payload and nested allocations; NULL is allowed.
FFI_EXPORT void loon_talon_result_free(LoonTalonResult* result);

#ifdef __cplusplus
}
#endif
