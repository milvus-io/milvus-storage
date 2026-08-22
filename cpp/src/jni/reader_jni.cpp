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
#include "milvus-storage/ffi_internal/record_batch_reader.h"
#include <arrow/c/abi.h>
#include <vector>

// ==================== JNI Reader Implementation ====================
//
// All JNI entry points must have C linkage so their exported symbols match
// the unmangled names `Java_<class>_<method>` that JNI looks up at runtime.
// Some of the functions below are also declared in `ffi_jni.h`'s `extern "C"`
// block (and thus get C linkage via the header), but newer additions like
// `recordBatchReaderNew`, `recordBatchReaderReadNext`, `recordBatchReaderDestroy`,
// `getChunkReader`, and `take` are not. An `extern "C"` wrapper around every
// definition below makes the linkage uniform and prevents silent mismatches
// that surface as `UnsatisfiedLinkError` only at runtime.
extern "C" {

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_readerNew(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong column_groups,
                                                                             jlong schema_ptr,
                                                                             jobjectArray needed_columns,
                                                                             jlong properties_ptr) {
  try {
    LoonColumnGroups* column_groups_ptr = reinterpret_cast<LoonColumnGroups*>(column_groups);
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck()) {
      return -1;
    }

    LoonReaderHandle reader_handle;
    LoonFFIResult result = loon_reader_new(column_groups_ptr, schema, columns, num_columns, properties, &reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return static_cast<jlong>(reader_handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

// ==================== Per-batch RecordBatchReader JNI ====================
//
// Alternative to getRecordBatchReader above. Mirrors Milvus's segcore
// ReadNext binding: caller pulls one RecordBatch at a time, each
// exported as a fresh ArrowArray+ArrowSchema pair. Required because
// Arrow Java's ArrowArrayStream-based reader shares a single
// VectorSchemaRoot across batches and ignores per-batch ArrowArray
// offset, causing duplicate reads when the underlying C++ reader emits
// RecordBatch::Slice results. See
// https://github.com/zilliztech/spark-milvus for the failing reproducer.

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderNew(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong reader_handle,
                                                                                        jstring predicate) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);
    const char* predicate_cstr = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;
    if (predicate != nullptr && predicate_cstr == nullptr) {
      return -1;
    }

    LoonRecordBatchReaderHandle rbr_handle = 0;
    LoonFFIResult result = loon_record_batch_reader_new(handle, predicate_cstr, &rbr_handle, nullptr);

    if (predicate_cstr) {
      env->ReleaseStringUTFChars(predicate, predicate_cstr);
    }

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return static_cast<jlong>(rbr_handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

// Reads the next batch into the caller-allocated ArrowArray + ArrowSchema
// pointed to by `array_addr` / `schema_addr`. Both pointers must reference
// zero-initialized structs allocated on the Java side (typically via
// `ArrowArray.allocateNew` + `ArrowSchema.allocateNew`).
//
// Returns true when a batch was produced (caller imports + releases the
// structs), false on EOF (structs' `release` fields are NULL).
JNIEXPORT jboolean JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderReadNext(
    JNIEnv* env, jobject obj, jlong rbr_handle, jlong array_addr, jlong schema_addr) {
  try {
    auto handle = static_cast<LoonRecordBatchReaderHandle>(rbr_handle);
    auto* out_array = reinterpret_cast<ArrowArray*>(array_addr);
    auto* out_schema = reinterpret_cast<ArrowSchema*>(schema_addr);

    // Arrow Java ignores ArrowArray.offset. Keep the JNI-specific copy in this
    // adapter while the public C FFI uses the standard zero-copy export.
    LoonFFIResult result =
        milvus_storage::ffi_internal::RecordBatchReaderReadNextForJava(handle, out_array, out_schema);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return JNI_FALSE;
    }

    // EOF contract: release == nullptr on both structs.
    return (out_array->release == nullptr) ? JNI_FALSE : JNI_TRUE;
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return JNI_FALSE;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderDestroy(JNIEnv* env,
                                                                                           jobject obj,
                                                                                           jlong rbr_handle) {
  try {
    loon_record_batch_reader_destroy(static_cast<LoonRecordBatchReaderHandle>(rbr_handle));
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getChunkReader(
    JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id, jobjectArray needed_columns) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck()) {
      return -1;
    }

    LoonChunkReaderHandle chunk_reader_handle;
    LoonFFIResult result = loon_get_chunk_reader(handle, static_cast<int64_t>(column_group_id), columns, num_columns,
                                                 &chunk_reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return static_cast<jlong>(chunk_reader_handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageReader_take(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong reader_handle,
                                                                             jlongArray row_indices,
                                                                             jlong parallelism,
                                                                             jobjectArray needed_columns) {
  try {
    if (row_indices == nullptr) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "rowIndices must not be null");
      return nullptr;
    }
    // Preserve the signed Java boundary. loon_take accepts size_t, where a
    // negative value would wrap and look like valid (but enormous) input.
    if (parallelism <= 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "parallelism must be > 0");
      return nullptr;
    }

    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);

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

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck()) {
      return nullptr;
    }

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    LoonFFIResult result =
        loon_take(handle, indices.data(), static_cast<size_t>(length), static_cast<size_t>(parallelism), columns,
                  num_columns, &arrays, &num_arrays, nullptr);

    FreeStringArray(env, columns, num_columns);

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

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong reader_handle) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);
    loon_reader_destroy(handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
  }
}

}  // extern "C"
