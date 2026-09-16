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
#include "jni_raii.h"
#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/buffer.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

using namespace milvus_storage::api;
using namespace milvus_storage;
using namespace milvus_storage::jni;

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
  int64_t batches = 0;
  int64_t copies = 0;
  int64_t copied_bytes = 0;
};

int64_t BufferBytes(const arrow::Array& array) {
  int64_t bytes = 0;
  for (const auto& buffer : array.data()->buffers) {
    if (buffer != nullptr)
      bytes += buffer->size();
  }
  return bytes;
}

// Importing each ArrowArray transfers its release callback, but the outer
// allocation still belongs to loon_take. This also frees unconsumed arrays
// after an import failure or a Java allocation failure.
struct TakeOutput {
  ArrowArray* arrays = nullptr;
  size_t count = 0;
  ArrowSchema schema{};

  ~TakeOutput() {
    if (schema.release != nullptr)
      schema.release(&schema);
    loon_free_chunk_arrays(arrays, count);
  }
};

void ThrowArgument(JNIEnv* env, const char* message) { Throw(env, "java/lang/IllegalArgumentException", message); }

void ThrowArrowStatus(JNIEnv* env, const arrow::Status& status) {
  std::string message = status.ToString();
  LoonFFIResult result{FFIErrorCodeFromExtendStatus(status), const_cast<char*>(message.c_str())};
  ThrowJavaExceptionFromFFIResult(env, &result);
}

bool PrepareTake(JNIEnv* env,
                 jlong reader,
                 jlongArray row_indices,
                 jlong parallelism,
                 jobjectArray needed_columns,
                 bool require_sorted,
                 TakeOutput* output) {
  if (reader == 0 || row_indices == nullptr || parallelism <= 0) {
    ThrowArgument(env, "reader and row indices are required; parallelism must be positive");
    return false;
  }
  jsize count = env->GetArrayLength(row_indices);
  if (count == 0) {
    ThrowArgument(env, "row indices must not be empty");
    return false;
  }
  std::vector<jlong> java_indices(static_cast<size_t>(count));
  env->GetLongArrayRegion(row_indices, 0, count, java_indices.data());
  if (env->ExceptionCheck())
    return false;
  std::vector<int64_t> indices(java_indices.begin(), java_indices.end());
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0 || (require_sorted && i > 0 && indices[i] <= indices[i - 1])) {
      ThrowArgument(env, "row indices must be nonnegative, sorted and unique");
      return false;
    }
  }
  size_t num_columns = 0;
  const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
  if (env->ExceptionCheck())
    return false;
  LoonFFIResult result =
      loon_take(static_cast<LoonReaderHandle>(reader), indices.data(), indices.size(), static_cast<size_t>(parallelism),
                columns, num_columns, &output->arrays, &output->count, &output->schema);
  FreeStringArray(env, columns, num_columns);
  return CheckResult(env, result);
}

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
            auto copied = concat_result.ValueOrDie();
            holder->copies += 1;
            holder->copied_bytes += BufferBytes(*copied);
            fresh_cols.push_back(std::move(copied));
          }
        }
        batch = arrow::RecordBatch::Make(batch->schema(), batch->num_rows(), fresh_cols);
      }

      auto export_status = arrow::ExportRecordBatch(*batch, out_array, out_schema);
      if (!export_status.ok()) {
        RETURN_ERROR(LOON_ARROW_ERROR, export_status.ToString());
      }
      holder->batches += 1;
    } else {  // batch == nullptr
      out_array->release = nullptr;
      out_schema->release = nullptr;
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
    if (env->ExceptionCheck())
      return 0;

    LoonReaderHandle reader_handle;
    LoonFFIResult result = loon_reader_new(column_groups_ptr, schema, columns, num_columns, properties, &reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    loon_ffi_free_result(&result);
    return static_cast<jlong>(reader_handle);
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to create reader: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
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
    if (env->ExceptionCheck())
      return 0;

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

    loon_ffi_free_result(&result);
    return static_cast<jlong>(rbr_handle);
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to open record batch reader: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
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

    loon_ffi_free_result(&result);
    // EOF contract: release == nullptr on both structs.
    return (out_array->release == nullptr) ? JNI_FALSE : JNI_TRUE;
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to read next record batch: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
    return JNI_FALSE;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderDestroy(JNIEnv* env,
                                                                                           jobject obj,
                                                                                           jlong rbr_handle) {
  try {
    loon_record_batch_reader_destroy(static_cast<LoonRecordBatchReaderHandle>(rbr_handle));
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to destroy record batch reader: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_getChunkReader(
    JNIEnv* env, jobject obj, jlong reader_handle, jlong column_group_id, jobjectArray needed_columns) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck())
      return 0;

    LoonChunkReaderHandle chunk_reader_handle;
    LoonFFIResult result = loon_get_chunk_reader(handle, static_cast<int64_t>(column_group_id), columns, num_columns,
                                                 &chunk_reader_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    loon_ffi_free_result(&result);
    return static_cast<jlong>(chunk_reader_handle);
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to get chunk reader: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageReader_take(
    JNIEnv* env, jobject, jlong reader, jlongArray row_indices, jlong parallelism, jobjectArray needed_columns) {
  return Guard<jlongArray>(env, nullptr, [&]() -> jlongArray {
    TakeOutput output;
    if (!PrepareTake(env, reader, row_indices, parallelism, needed_columns, false, &output))
      return nullptr;
    std::vector<jlong> addresses(output.count);
    for (size_t i = 0; i < output.count; ++i) addresses[i] = reinterpret_cast<jlong>(&output.arrays[i]);
    jlongArray result = env->NewLongArray(static_cast<jsize>(output.count));
    if (result == nullptr)
      return nullptr;
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(output.count), addresses.data());
    if (env->ExceptionCheck())
      return nullptr;
    output.arrays = nullptr;
    output.count = 0;
    return result;
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_freeTakeRows(JNIEnv* env,
                                                                               jobject,
                                                                               jlongArray addresses) {
  if (addresses == nullptr)
    return;
  GuardVoid(env, [&] {
    jsize count = env->GetArrayLength(addresses);
    if (count == 0)
      return;
    std::vector<jlong> values(static_cast<size_t>(count));
    env->GetLongArrayRegion(addresses, 0, count, values.data());
    if (env->ExceptionCheck())
      return;
    auto* arrays = reinterpret_cast<ArrowArray*>(values.front());
    if (arrays == nullptr) {
      ThrowArgument(env, "take arrays must be the unmodified result of one take call");
      return;
    }
    for (jsize i = 0; i < count; ++i) {
      if (values[i] != reinterpret_cast<jlong>(&arrays[i])) {
        ThrowArgument(env, "take arrays must be the unmodified result of one take call");
        return;
      }
    }
    loon_free_chunk_arrays(arrays, static_cast<size_t>(count));
  });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageReader_takeRecordBatchReader(
    JNIEnv* env, jobject, jlong reader, jlongArray row_indices, jlong parallelism, jobjectArray needed_columns) {
  return Guard<jlong>(env, 0, [&]() -> jlong {
    TakeOutput output;
    if (!PrepareTake(env, reader, row_indices, parallelism, needed_columns, true, &output))
      return 0;
    if (output.arrays == nullptr || output.count == 0 || output.schema.release == nullptr) {
      ThrowArrowStatus(env, arrow::Status::Invalid("loon_take returned no batches for nonempty indices"));
      return 0;
    }
    auto imported_schema = arrow::ImportSchema(&output.schema);
    if (!imported_schema.ok()) {
      ThrowArrowStatus(env, imported_schema.status());
      return 0;
    }
    auto schema = imported_schema.MoveValueUnsafe();
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    batches.reserve(output.count);
    int64_t rows = 0;
    for (size_t i = 0; i < output.count; ++i) {
      auto imported = arrow::ImportRecordBatch(&output.arrays[i], schema);
      if (!imported.ok()) {
        ThrowArrowStatus(env, imported.status());
        return 0;
      }
      auto batch = imported.MoveValueUnsafe();
      rows += batch->num_rows();
      batches.push_back(std::move(batch));
    }
    if (rows != env->GetArrayLength(row_indices)) {
      ThrowArrowStatus(env, arrow::Status::Invalid("loon_take returned a different number of rows than requested"));
      return 0;
    }
    auto batch_reader = arrow::RecordBatchReader::Make(std::move(batches), schema);
    if (!batch_reader.ok()) {
      ThrowArrowStatus(env, batch_reader.status());
      return 0;
    }
    auto holder = std::make_unique<RecordBatchReaderHolder>();
    holder->reader = batch_reader.MoveValueUnsafe();
    return reinterpret_cast<jlong>(holder.release());
  });
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageReader_recordBatchReaderStats(JNIEnv* env,
                                                                                               jobject,
                                                                                               jlong handle) {
  if (handle == 0) {
    ThrowArgument(env, "record batch reader must not be null");
    return nullptr;
  }
  auto* holder = reinterpret_cast<RecordBatchReaderHolder*>(handle);
  jlong values[3] = {holder->batches, holder->copies, holder->copied_bytes};
  jlongArray result = env->NewLongArray(3);
  if (result != nullptr)
    env->SetLongArrayRegion(result, 0, 3, values);
  return result;
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageReader_readerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong reader_handle) {
  try {
    LoonReaderHandle handle = static_cast<LoonReaderHandle>(reader_handle);
    loon_reader_destroy(handle);
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to destroy reader: " + std::string(e.what());
    Throw(env, "java/lang/RuntimeException", error_msg.c_str());
    return;
  }
}

}  // extern "C"
