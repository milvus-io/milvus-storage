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

// ==================== JNI Reader Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_readerNew(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong column_groups,
                                                                             jlong schema_ptr,
                                                                             jobjectArray needed_columns,
                                                                             jlong properties_ptr) {
  try {
    ColumnGroupsHandle column_groups_handle = static_cast<ColumnGroupsHandle>(column_groups);
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

    ReaderHandle reader_handle;
    FFIResult result = reader_new(column_groups_handle, schema, columns, num_columns, properties, &reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return -1;
    }

    return static_cast<jlong>(reader_handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to create reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getRecordBatchReader(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong reader_handle,
                                                                                        jstring predicate) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);
    const char* predicate_cstr = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;

    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(malloc(sizeof(ArrowArrayStream)));
    FFIResult result = get_record_batch_reader(handle, predicate_cstr, stream);

    if (predicate_cstr) {
      env->ReleaseStringUTFChars(predicate, predicate_cstr);
    }

    if (!IsSuccess(&result)) {
      if (stream->release != nullptr) {
        stream->release(stream);
      }
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return -1;
    }

    return reinterpret_cast<jlong>(stream);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get record batch reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getChunkReader(JNIEnv* env,
                                                                                  jobject obj,
                                                                                  jlong reader_handle,
                                                                                  jlong column_group_id) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);

    ChunkReaderHandle chunk_reader_handle;
    FFIResult result = get_chunk_reader(handle, static_cast<int64_t>(column_group_id), &chunk_reader_handle);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return -1;
    }

    return static_cast<jlong>(chunk_reader_handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get chunk reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageReader_take(
    JNIEnv* env, jobject obj, jlong reader_handle, jlongArray row_indices, jlong parallelism) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    jlong* indices_array = env->GetLongArrayElements(row_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    FFIResult result = take(handle, indices.data(), static_cast<size_t>(length), static_cast<int64_t>(parallelism),
                            &arrays, &num_arrays);

    env->ReleaseLongArrayElements(row_indices, indices_array, JNI_ABORT);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
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
    std::string error_msg = "Failed to take rows: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong reader_handle) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);
    reader_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}
