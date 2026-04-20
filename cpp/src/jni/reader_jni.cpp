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
#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/reader.h"
#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

using namespace milvus_storage::api;
using namespace milvus_storage;

// ==================== Per-batch RecordBatchReader (JNI-only helpers) ====================
//
// These C helpers back the `loon_record_batch_reader_*` declarations in
// `ffi_jni.h`. They are deliberately defined in the JNI translation unit
// (linked into `libmilvus-storage-jni.so` only) rather than in
// `src/ffi/reader_c.cpp` because the offset-0 materialization they perform
// is a workaround for Arrow Java's C Data importer, which ignores
// `ArrowArray.offset`. Non-JVM consumers already handle sliced batches
// correctly via `loon_get_record_batch_reader` and should not see the
// memory copy.
//
// Lifecycle:
//   - `loon_record_batch_reader_new` opens a handle owning a
//     `shared_ptr<arrow::RecordBatchReader>`.
//   - `loon_record_batch_reader_read_next` fills caller-owned
//     ArrowArray/ArrowSchema structs; on EOF the `release` fields are
//     left NULL.
//   - `loon_record_batch_reader_destroy` drops the handle.

namespace {

struct RecordBatchReaderHolder {
  std::shared_ptr<arrow::RecordBatchReader> reader;
};

}  // namespace

extern "C" LoonFFIResult loon_record_batch_reader_new(LoonReaderHandle reader,
                                                      const char* predicate,
                                                      LoonRecordBatchReaderHandle* out_handle) {
  if (!reader || !out_handle) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: reader and out_handle must not be null");
  }

  try {
    auto* cpp_reader = reinterpret_cast<Reader*>(reader);
    std::string predicate_str = predicate ? predicate : "";

    auto result = cpp_reader->get_record_batch_reader(predicate_str);
    if (!result.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, result.status().ToString());
    }

    auto* holder = new RecordBatchReaderHolder{result.ValueOrDie()};
    *out_handle = reinterpret_cast<LoonRecordBatchReaderHandle>(holder);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

extern "C" LoonFFIResult loon_record_batch_reader_read_next(LoonRecordBatchReaderHandle handle,
                                                            struct ArrowArray* out_array,
                                                            struct ArrowSchema* out_schema) {
  if (!handle || !out_array || !out_schema) {
    RETURN_ERROR(LOON_INVALID_ARGS, "Invalid arguments: handle, out_array, out_schema must not be null");
  }

  try {
    auto* holder = reinterpret_cast<RecordBatchReaderHolder*>(handle);
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = holder->reader->ReadNext(&batch);
    if (!status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, status.ToString());
    }

    // PackedRecordBatchReader::ReadNext can hand back a RecordBatch whose
    // column arrays carry a non-zero `offset` — this happens whenever the
    // underlying chunk is larger than min_rows and the remainder is kept
    // in the queue via `rb->Slice(min_rows)` (see reader.cpp). ArrowArray's
    // C Data Interface specifies consumers must honour `offset`, but Arrow
    // Java's `Data.importVectorSchemaRoot` ignores it. Materialize sliced
    // columns into fresh offset=0 arrays via arrow::Concatenate (copies
    // only the slice range). Non-sliced columns pass through unchanged.
    if (batch != nullptr) {
      bool has_sliced_column = false;
      for (int i = 0; i < batch->num_columns(); ++i) {
        if (batch->column(i)->offset() != 0) {
          has_sliced_column = true;
          break;
        }
      }
      if (has_sliced_column) {
        std::vector<std::shared_ptr<arrow::Array>> fresh_cols;
        fresh_cols.reserve(batch->num_columns());
        for (int i = 0; i < batch->num_columns(); ++i) {
          auto col = batch->column(i);
          if (col->offset() == 0) {
            fresh_cols.push_back(col);
          } else {
            auto concat_result = arrow::Concatenate({col}, arrow::default_memory_pool());
            if (!concat_result.ok()) {
              RETURN_ERROR(LOON_ARROW_ERROR, concat_result.status().ToString());
            }
            fresh_cols.push_back(concat_result.ValueOrDie());
          }
        }
        batch = arrow::RecordBatch::Make(batch->schema(), batch->num_rows(), fresh_cols);
      }
    }

    if (batch == nullptr) {
      out_array->release = nullptr;
      out_schema->release = nullptr;
      RETURN_SUCCESS();
    }

    auto export_status = arrow::ExportRecordBatch(*batch, out_array, out_schema);
    if (!export_status.ok()) {
      RETURN_ERROR(LOON_ARROW_ERROR, export_status.ToString());
    }

    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

extern "C" void loon_record_batch_reader_destroy(LoonRecordBatchReaderHandle handle) {
  if (!handle)
    return;
  delete reinterpret_cast<RecordBatchReaderHolder*>(handle);
}

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

    LoonReaderHandle reader_handle;
    LoonFFIResult result = loon_reader_new(column_groups_ptr, schema, columns, num_columns, properties, &reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
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
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);
    const char* predicate_cstr = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;

    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(calloc(1, sizeof(ArrowArrayStream)));
    LoonFFIResult result = loon_get_record_batch_reader(handle, predicate_cstr, stream);

    if (predicate_cstr) {
      env->ReleaseStringUTFChars(predicate, predicate_cstr);
    }

    if (!loon_ffi_is_success(&result)) {
      if (stream->release != nullptr) {
        stream->release(stream);
      }
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
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

    LoonRecordBatchReaderHandle rbr_handle = 0;
    LoonFFIResult result = loon_record_batch_reader_new(handle, predicate_cstr, &rbr_handle);

    if (predicate_cstr) {
      env->ReleaseStringUTFChars(predicate, predicate_cstr);
    }

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return static_cast<jlong>(rbr_handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to open record batch reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
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

    LoonFFIResult result = loon_record_batch_reader_read_next(handle, out_array, out_schema);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return JNI_FALSE;
    }

    // EOF contract: release == nullptr on both structs.
    return (out_array->release == nullptr) ? JNI_FALSE : JNI_TRUE;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to read next record batch: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return JNI_FALSE;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderDestroy(JNIEnv* env,
                                                                                           jobject obj,
                                                                                           jlong rbr_handle) {
  try {
    loon_record_batch_reader_destroy(static_cast<LoonRecordBatchReaderHandle>(rbr_handle));
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy record batch reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getChunkReader(
    JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id, jobjectArray needed_columns) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

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
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get chunk reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
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
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);

    jsize length = env->GetArrayLength(row_indices);
    jlong* indices_array = env->GetLongArrayElements(row_indices, nullptr);

    std::vector<int64_t> indices(length);
    for (jsize i = 0; i < length; ++i) {
      indices[i] = static_cast<int64_t>(indices_array[i]);
    }

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

    ArrowArray* arrays = nullptr;
    size_t num_arrays = 0;
    LoonFFIResult result =
        loon_take(handle, indices.data(), static_cast<size_t>(length), static_cast<int64_t>(parallelism), columns,
                  num_columns, &arrays, &num_arrays, nullptr);

    FreeStringArray(env, columns, num_columns);
    env->ReleaseLongArrayElements(row_indices, indices_array, JNI_ABORT);

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
    std::string error_msg = "Failed to take rows: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong reader_handle) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);
    loon_reader_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy reader: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

}  // extern "C"
