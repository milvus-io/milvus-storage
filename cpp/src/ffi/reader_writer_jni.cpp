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

#include "milvus-storage/ffi_jni.h"
#include "milvus-storage/ffi_c.h"
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <cstring>
#include <memory>
#include <vector>

extern "C" {
JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageBridge_CallTheCFunction(JNIEnv* env, jobject obj) {
  // Implement the function to test whether JNI is working
  return 0;
}
}

// ==================== JNI Utility Functions ====================

void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct ffi_result* result) {
  if (IsSuccess(const_cast<FFIResult*>(result))) {
    return;
  }

  const char* message = GetErrorMessage(const_cast<FFIResult*>(result));
  const char* exception_class = "java/lang/RuntimeException";

  switch (result->err_code) {
    case LOON_INVALID_ARGS:
      exception_class = "java/lang/IllegalArgumentException";
      break;
    case LOON_MEMORY_ERROR:
      exception_class = "java/lang/OutOfMemoryError";
      break;
    case LOON_ARROW_ERROR:
    case LOON_LOGICAL_ERROR:
    case LOON_GOT_EXCEPTION:
    case LOON_UNREACHABLE_ERROR:
    case LOON_INVALID_PROPERTIES:
    default:
      exception_class = "java/lang/RuntimeException";
      break;
  }

  jclass exc_class = env->FindClass(exception_class);
  if (exc_class != nullptr) {
    env->ThrowNew(exc_class, message ? message : "Unknown error");
  }
}

jobjectArray ConvertToJavaStringArray(JNIEnv* env, const char* const* strings, size_t count) {
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(count), string_class, nullptr);

  for (size_t i = 0; i < count; ++i) {
    jstring str = env->NewStringUTF(strings[i]);
    env->SetObjectArrayElement(result, static_cast<jsize>(i), str);
    env->DeleteLocalRef(str);
  }

  return result;
}

const char** ConvertFromJavaStringArray(JNIEnv* env, jobjectArray java_array, size_t* out_count) {
  if (java_array == nullptr) {
    *out_count = 0;
    return nullptr;
  }

  jsize length = env->GetArrayLength(java_array);
  *out_count = static_cast<size_t>(length);

  const char** strings = static_cast<const char**>(malloc(sizeof(char*) * length));
  for (jsize i = 0; i < length; ++i) {
    jstring jstr = static_cast<jstring>(env->GetObjectArrayElement(java_array, i));
    const char* str = env->GetStringUTFChars(jstr, nullptr);
    strings[i] = strdup(str);
    env->ReleaseStringUTFChars(jstr, str);
    env->DeleteLocalRef(jstr);
  }

  return strings;
}

void FreeStringArray(JNIEnv* env, const char** strings, size_t count) {
  if (strings != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      free(const_cast<char*>(strings[i]));
    }
    free(strings);
  }
}

// ==================== JNI Properties Implementation ====================

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageProperties_allocateProperties(JNIEnv* env, jobject obj) {
  try {
    Properties* properties = static_cast<Properties*>(malloc(sizeof(Properties)));
    if (properties == nullptr) {
      jclass exc_class = env->FindClass("java/lang/OutOfMemoryError");
      env->ThrowNew(exc_class, "Failed to allocate memory for Properties");
      return 0;
    }

    // Initialize the properties structure
    properties->properties = nullptr;
    properties->count = 0;

    return reinterpret_cast<jlong>(properties);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to allocate properties");
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageProperties_createProperties(JNIEnv* env, jobject obj, jobject java_map, jlong properties_ptr) {
  try {
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    jclass map_class = env->GetObjectClass(java_map);
    jmethodID entry_set_method = env->GetMethodID(map_class, "entrySet", "()Ljava/util/Set;");
    jobject entry_set = env->CallObjectMethod(java_map, entry_set_method);

    jclass set_class = env->GetObjectClass(entry_set);
    jmethodID to_array_method = env->GetMethodID(set_class, "toArray", "()[Ljava/lang/Object;");
    jobjectArray entries = static_cast<jobjectArray>(env->CallObjectMethod(entry_set, to_array_method));

    jsize num_entries = env->GetArrayLength(entries);

    std::vector<const char*> keys, values;
    std::vector<std::string> key_storage, value_storage;

    for (jsize i = 0; i < num_entries; ++i) {
      jobject entry = env->GetObjectArrayElement(entries, i);
      jclass entry_class = env->GetObjectClass(entry);

      jmethodID get_key_method = env->GetMethodID(entry_class, "getKey", "()Ljava/lang/Object;");
      jmethodID get_value_method = env->GetMethodID(entry_class, "getValue", "()Ljava/lang/Object;");

      jstring key_jstr = static_cast<jstring>(env->CallObjectMethod(entry, get_key_method));
      jstring value_jstr = static_cast<jstring>(env->CallObjectMethod(entry, get_value_method));

      const char* key_cstr = env->GetStringUTFChars(key_jstr, nullptr);
      const char* value_cstr = env->GetStringUTFChars(value_jstr, nullptr);

      key_storage.emplace_back(key_cstr);
      value_storage.emplace_back(value_cstr);

      env->ReleaseStringUTFChars(key_jstr, key_cstr);
      env->ReleaseStringUTFChars(value_jstr, value_cstr);
      env->DeleteLocalRef(key_jstr);
      env->DeleteLocalRef(value_jstr);
      env->DeleteLocalRef(entry);
    }

    for (size_t i = 0; i < key_storage.size(); ++i) {
      keys.push_back(key_storage[i].c_str());
      values.push_back(value_storage[i].c_str());
    }

    FFIResult result = properties_create(keys.data(), values.data(), keys.size(), properties);
    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
    }

    env->DeleteLocalRef(entries);
    env->DeleteLocalRef(entry_set);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to create properties");
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageProperties_freeProperties(JNIEnv* env, jobject obj, jlong properties_ptr) {
  Properties* properties = reinterpret_cast<Properties*>(properties_ptr);
  if (properties != nullptr) {
    properties_free(properties);
  }
}

// ==================== JNI Writer Implementation ====================

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageWriter_writerNew(JNIEnv* env, jobject obj, jstring base_path, jlong schema_ptr, jlong properties_ptr) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    WriterHandle writer_handle;
    FFIResult result = writer_new(base_path_cstr, schema, properties, &writer_handle);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return static_cast<jlong>(writer_handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to create writer");
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageWriter_writerWrite(JNIEnv* env, jobject obj, jlong writer_handle, jlong array_ptr) {
  try {
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);
    ArrowArray* array = reinterpret_cast<ArrowArray*>(array_ptr);

    FFIResult result = writer_write(handle, array);
    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
    }
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to write to writer");
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageWriter_writerFlush(JNIEnv* env, jobject obj, jlong writer_handle) {
  try {
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);

    FFIResult result = writer_flush(handle);
    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
    }
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to flush writer");
  }
}

JNIEXPORT jstring JNICALL
Java_io_milvus_storage_MilvusStorageWriter_writerClose(JNIEnv* env, jobject obj, jlong writer_handle) {
  try {
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);

    char* manifest = nullptr;
    size_t manifest_size = 0;
    FFIResult result = writer_close(handle, &manifest, &manifest_size);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return nullptr;
    }

    jstring java_manifest = env->NewStringUTF(manifest);
    free_manifest(manifest);

    return java_manifest;
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to close writer");
    return nullptr;
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageWriter_writerDestroy(JNIEnv* env, jobject obj, jlong writer_handle) {
  try {
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);
    writer_destroy(handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to destroy writer");
  }
}

// ==================== JNI Reader Implementation ====================

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageReader_readerNew(JNIEnv* env, jobject obj, jstring manifest, jlong schema_ptr, jobjectArray needed_columns, jlong properties_ptr) {
  try {
    const char* manifest_cstr = env->GetStringUTFChars(manifest, nullptr);
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

    ReaderHandle reader_handle;
    FFIResult result = reader_new(const_cast<char*>(manifest_cstr), schema, columns, num_columns, properties, &reader_handle);

    env->ReleaseStringUTFChars(manifest, manifest_cstr);
    FreeStringArray(env, columns, num_columns);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return static_cast<jlong>(reader_handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to create reader");
    return 0;
  }
}

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageReader_getRecordBatchReader(JNIEnv* env, jobject obj, jlong reader_handle, jstring predicate, jlong batch_size, jlong buffer_size) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);
    const char* predicate_cstr = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;

    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(malloc(sizeof(ArrowArrayStream)));
    FFIResult result = get_record_batch_reader(handle, predicate_cstr, static_cast<int64_t>(batch_size), static_cast<int64_t>(buffer_size), stream);

    if (predicate_cstr) {
      env->ReleaseStringUTFChars(predicate, predicate_cstr);
    }

    if (!IsSuccess(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return reinterpret_cast<jlong>(stream);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to get record batch reader");
    return 0;
  }
}

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageReader_getChunkReader(JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);

    ChunkReaderHandle chunk_reader_handle;
    FFIResult result = get_chunk_reader(handle, static_cast<int64_t>(column_group_id), &chunk_reader_handle);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return static_cast<jlong>(chunk_reader_handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to get chunk reader");
    return 0;
  }
}

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageReader_take(JNIEnv* env, jobject obj, jlong reader_handle, jlongArray row_indices, jlong parallelism) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    jlong* indices_array = env->GetLongArrayElements(row_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    ArrowArray* array = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray)));
    FFIResult result = take(handle, indices.data(), static_cast<size_t>(length), static_cast<int64_t>(parallelism), array);

    env->ReleaseLongArrayElements(row_indices, indices_array, JNI_ABORT);

    if (!IsSuccess(&result)) {
      free(array);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return reinterpret_cast<jlong>(array);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to take rows");
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env, jobject obj, jlong reader_handle) {
  try {
    ReaderHandle handle = static_cast<ReaderHandle>(reader_handle);
    reader_destroy(handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to destroy reader");
  }
}

// ==================== JNI ChunkReader Implementation ====================

JNIEXPORT jlongArray JNICALL
Java_io_milvus_storage_MilvusStorageChunkReader_getChunkIndices(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray row_indices) {
  try {
    ChunkReaderHandle handle = static_cast<ChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    jlong* indices_array = env->GetLongArrayElements(row_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    int64_t* chunk_indices = nullptr;
    size_t num_chunk_indices = 0;
    FFIResult result = get_chunk_indices(handle, indices.data(), static_cast<size_t>(length), &chunk_indices, &num_chunk_indices);

    env->ReleaseLongArrayElements(row_indices, indices_array, JNI_ABORT);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return nullptr;
    }

    jlongArray java_chunk_indices = env->NewLongArray(static_cast<jsize>(num_chunk_indices));
    jlong* java_indices_array = env->GetLongArrayElements(java_chunk_indices, nullptr);

    for (size_t i = 0; i < num_chunk_indices; ++i) {
      java_indices_array[i] = static_cast<jlong>(chunk_indices[i]);
    }

    env->ReleaseLongArrayElements(java_chunk_indices, java_indices_array, 0);
    free(chunk_indices);

    return java_chunk_indices;
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to get chunk indices");
    return nullptr;
  }
}

JNIEXPORT jlong JNICALL
Java_io_milvus_storage_MilvusStorageChunkReader_getChunk(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlong chunk_index) {
  try {
    ChunkReaderHandle handle = static_cast<ChunkReaderHandle>(chunk_reader_handle);

    ArrowArray* array = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray)));
    FFIResult result = get_chunk(handle, static_cast<int64_t>(chunk_index), array);

    if (!IsSuccess(&result)) {
      free(array);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return 0;
    }

    return reinterpret_cast<jlong>(array);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to get chunk");
    return 0;
  }
}

JNIEXPORT jlongArray JNICALL
Java_io_milvus_storage_MilvusStorageChunkReader_getChunks(JNIEnv* env, jobject obj, jlong chunk_reader_handle, jlongArray chunk_indices, jlong parallelism) {
  try {
    ChunkReaderHandle handle = static_cast<ChunkReaderHandle>(chunk_reader_handle);

    jsize length = env->GetArrayLength(chunk_indices);
    jlong* indices_array = env->GetLongArrayElements(chunk_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    FFIResult result = get_chunks(handle, indices.data(), static_cast<size_t>(length), static_cast<int64_t>(parallelism), &arrays, &num_arrays);

    env->ReleaseLongArrayElements(chunk_indices, indices_array, JNI_ABORT);

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
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to get chunks");
    return nullptr;
  }
}

JNIEXPORT void JNICALL
Java_io_milvus_storage_MilvusStorageChunkReader_chunkReaderDestroy(JNIEnv* env, jobject obj, jlong chunk_reader_handle) {
  try {
    ChunkReaderHandle handle = static_cast<ChunkReaderHandle>(chunk_reader_handle);
    chunk_reader_destroy(handle);
  } catch (...) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    env->ThrowNew(exc_class, "Failed to destroy chunk reader");
  }
}