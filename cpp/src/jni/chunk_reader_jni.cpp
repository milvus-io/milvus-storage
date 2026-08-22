// Copyright 2025 Zilliz
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

#include "milvus-storage/ffi_jni.h"
#include "milvus-storage/ffi_c.h"
#include <arrow/c/abi.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// ==================== JNI ChunkReader Implementation ====================

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunkIndices(JNIEnv* env,
                                                                                             jobject obj,
                                                                                             jlong chunk_reader_handle,
                                                                                             jlongArray row_indices) {
  try {
    if (row_indices == nullptr) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "rowIndices must not be null");
      return nullptr;
    }

    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    if (length == 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "rowIndices must not be empty");
      return nullptr;
    }
    std::vector<jlong> java_indices(static_cast<size_t>(length));
    env->GetLongArrayRegion(row_indices, 0, length, java_indices.data());
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    std::vector<int64_t> indices(static_cast<size_t>(length));
    for (jsize i = 0; i < length; ++i) {
      indices[static_cast<size_t>(i)] = static_cast<int64_t>(java_indices[static_cast<size_t>(i)]);
    }

    std::unique_ptr<int64_t, decltype(&loon_free_chunk_indices)> chunk_indices_guard(nullptr, &loon_free_chunk_indices);
    int64_t* chunk_indices = nullptr;
    size_t num_chunk_indices = 0;
    LoonFFIResult result =
        loon_get_chunk_indices(handle, indices.data(), static_cast<size_t>(length), &chunk_indices, &num_chunk_indices);
    chunk_indices_guard.reset(chunk_indices);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    jlongArray java_chunk_indices = env->NewLongArray(static_cast<jsize>(num_chunk_indices));
    if (java_chunk_indices == nullptr) {
      return nullptr;
    }
    std::vector<jlong> java_chunk_values(static_cast<size_t>(num_chunk_indices));
    for (size_t i = 0; i < num_chunk_indices; ++i) {
      java_chunk_values[i] = static_cast<jlong>(chunk_indices_guard.get()[i]);
    }
    env->SetLongArrayRegion(java_chunk_indices, 0, static_cast<jsize>(num_chunk_indices), java_chunk_values.data());
    if (env->ExceptionCheck()) {
      return nullptr;
    }

    return java_chunk_indices;
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return nullptr;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunk(JNIEnv* env,
                                                                                 jobject obj,
                                                                                 jlong chunk_reader_handle,
                                                                                 jlong chunk_index) {
  try {
    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);

    ArrowArray* array = static_cast<ArrowArray*>(calloc(1, sizeof(ArrowArray)));
    if (array == nullptr) {
      ThrowJavaException(env, "java/lang/RuntimeException", "Unexpected native allocation failure for ArrowArray");
      return -1;
    }
    LoonFFIResult result = loon_get_chunk(handle, static_cast<int64_t>(chunk_index), array, nullptr);

    if (!loon_ffi_is_success(&result)) {
      if (array->release != nullptr) {
        array->release(array);
      }
      free(array);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return reinterpret_cast<jlong>(array);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunks(
    JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray chunk_indices, jlong parallelism) {
  try {
    if (chunk_indices == nullptr) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "chunkIndices must not be null");
      return nullptr;
    }
    // The C ABI takes size_t, so a negative Java value would otherwise wrap
    // to a huge positive number and bypass loon_get_chunks' zero check.
    if (parallelism <= 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "parallelism must be > 0");
      return nullptr;
    }

    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(chunk_indices);
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    if (length == 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "chunkIndices must not be empty");
      return nullptr;
    }
    std::vector<jlong> java_indices(static_cast<size_t>(length));
    env->GetLongArrayRegion(chunk_indices, 0, length, java_indices.data());
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    std::vector<int64_t> indices(static_cast<size_t>(length));
    for (jsize i = 0; i < length; ++i) {
      indices[static_cast<size_t>(i)] = static_cast<int64_t>(java_indices[static_cast<size_t>(i)]);
    }

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    LoonFFIResult result = loon_get_chunks(handle, indices.data(), static_cast<size_t>(length),
                                           static_cast<size_t>(parallelism), &arrays, &num_arrays, nullptr);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    jlongArray java_arrays = env->NewLongArray(static_cast<jsize>(num_arrays));
    if (java_arrays == nullptr) {
      loon_free_chunk_arrays(arrays, num_arrays);
      return nullptr;
    }
    for (size_t i = 0; i < num_arrays; ++i) {
      const jlong address = reinterpret_cast<jlong>(&arrays[i]);
      env->SetLongArrayRegion(java_arrays, static_cast<jsize>(i), 1, &address);
      if (env->ExceptionCheck()) {
        loon_free_chunk_arrays(arrays, num_arrays);
        return nullptr;
      }
    }

    return java_arrays;
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return nullptr;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_chunkReaderDestroy(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong chunk_reader_handle) {
  try {
    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);
    loon_chunk_reader_destroy(handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
  }
}
