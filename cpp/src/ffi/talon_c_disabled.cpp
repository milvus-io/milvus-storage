// Copyright 2026 Zilliz. Licensed under the Apache License, Version 2.0.
// Stubs for builds with WITH_EXTERNAL_TABLE_FORMATS=OFF. Callers link against the
// same C API and get an error if they try to use Talon.
#include "milvus-storage/ffi_talon_c.h"

#include "milvus-storage/ffi_internal/result.h"

extern "C" {

#define TALON_DISABLED() \
  RETURN_ERROR(LOON_INVALID_ARGS, "Talon is not available: this build has WITH_EXTERNAL_TABLE_FORMATS=OFF")

LoonFFIResult loon_talon_read(LoonTalonClientHandle, const char*, uint64_t, uint64_t, const char*, const uint64_t*, uint8_t*, const LoonTalonRequestOptions*, LoonTalonResult**) { TALON_DISABLED(); }

LoonFFIResult loon_talon_stat(LoonTalonClientHandle, const char*, const LoonTalonRequestOptions*, LoonTalonResult**) { TALON_DISABLED(); }

LoonFFIResult loon_talon_load(LoonTalonClientHandle, const LoonTalonLoadRequest*, const LoonTalonRequestOptions*, LoonTalonResult**) { TALON_DISABLED(); }

LoonFFIResult loon_talon_batch_load(LoonTalonClientHandle, const LoonTalonLoadRequest*, size_t, const LoonTalonRequestOptions*, LoonTalonResult**) { TALON_DISABLED(); }

LoonFFIResult loon_talon_read_async(LoonAsyncContextHandle, LoonTalonClientHandle, const char*, uint64_t, uint64_t, const char*, const uint64_t*, uint8_t*, const LoonTalonRequestOptions*, LoonTalonCallback, uintptr_t, LoonAsyncHandle*) { TALON_DISABLED(); }

LoonFFIResult loon_talon_stat_async(LoonAsyncContextHandle, LoonTalonClientHandle, const char*, const LoonTalonRequestOptions*, LoonTalonCallback, uintptr_t, LoonAsyncHandle*) { TALON_DISABLED(); }

LoonFFIResult loon_talon_load_async(LoonAsyncContextHandle, LoonTalonClientHandle, const LoonTalonLoadRequest*, const LoonTalonRequestOptions*, LoonTalonCallback, uintptr_t, LoonAsyncHandle*) { TALON_DISABLED(); }

LoonFFIResult loon_talon_batch_load_async(LoonAsyncContextHandle, LoonTalonClientHandle, const LoonTalonLoadRequest*, size_t, const LoonTalonRequestOptions*, LoonTalonCallback, uintptr_t, LoonAsyncHandle*) { TALON_DISABLED(); }

LoonFFIResult loon_talon_client_create(const LoonProperties*, LoonTalonClientHandle*) { TALON_DISABLED(); }

void loon_talon_client_destroy(LoonTalonClientHandle) {}

void loon_talon_result_free(LoonTalonResult*) {}

}  // extern "C"
