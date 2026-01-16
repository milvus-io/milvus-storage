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
#include <string>
#include <vector>

// ==================== JNI ChunkReader Implementation ====================

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunkIndices(JNIEnv* env,
                                                                                             jobject obj,
                                                                                             jlong chunk_reader_handle,
                                                                                             jlongArray row_indices) {
  try {
    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    jlong* indices_array = env->GetLongArrayElements(row_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    int64_t* chunk_indices = nullptr;
    size_t num_chunk_indices = 0;
    LoonFFIResult result =
        loon_get_chunk_indices(handle, indices.data(), static_cast<size_t>(length), &chunk_indices, &num_chunk_indices);

    env->ReleaseLongArrayElements(row_indices, indices_array, JNI_ABORT);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    jlongArray java_chunk_indices = env->NewLongArray(static_cast<jsize>(num_chunk_indices));
    jlong* java_indices_array = env->GetLongArrayElements(java_chunk_indices, nullptr);

    for (size_t i = 0; i < num_chunk_indices; ++i) {
      java_indices_array[i] = static_cast<jlong>(chunk_indices[i]);
    }

    env->ReleaseLongArrayElements(java_chunk_indices, java_indices_array, 0);
    loon_free_chunk_indices(chunk_indices);

    return java_chunk_indices;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get chunk indices: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
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
    LoonFFIResult result = loon_get_chunk(handle, static_cast<int64_t>(chunk_index), array);

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
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get chunk: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_getChunks(
    JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray chunk_indices, jlong parallelism) {
  try {
    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(chunk_indices);
    jlong* indices_array = env->GetLongArrayElements(chunk_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    LoonFFIResult result = loon_get_chunks(handle, indices.data(), static_cast<size_t>(length),
                                           static_cast<int64_t>(parallelism), &arrays, &num_arrays);

    env->ReleaseLongArrayElements(chunk_indices, indices_array, JNI_ABORT);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    jlongArray java_arrays = env->NewLongArray(static_cast<jsize>(num_arrays));
    jlong* java_arrays_ptr = env->GetLongArrayElements(java_arrays, nullptr);

    for (size_t i = 0; i < num_arrays; ++i) {
      java_arrays_ptr[i] = reinterpret_cast<jlong>(&arrays[i]);
    }

    env->ReleaseLongArrayElements(java_arrays, java_arrays_ptr, 0);

    return java_arrays;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get chunks: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageChunkReader_chunkReaderDestroy(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong chunk_reader_handle) {
  try {
    LoonChunkReaderHandle handle = static_cast<LoonChunkReaderHandle>(chunk_reader_handle);
    loon_chunk_reader_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy chunk reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}
