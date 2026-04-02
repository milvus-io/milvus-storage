// Copyright 2024 Zilliz
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
#include <string>
#include <vector>

// ==================== JNI SegmentReader Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderOpen(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jstring segment_path,
                                                                                     jlong version,
                                                                                     jlong schema_ptr,
                                                                                     jobjectArray needed_columns,
                                                                                     jlongArray lob_field_ids,
                                                                                     jobjectArray lob_base_paths,
                                                                                     jlongArray lob_inline_thresholds,
                                                                                     jlongArray lob_max_file_bytes,
                                                                                     jlongArray lob_flush_thresholds,
                                                                                     jlong properties_ptr) {
  try {
    const char* seg_path = env->GetStringUTFChars(segment_path, nullptr);
    auto* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    auto* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    // Convert needed columns
    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

    // Build LOB configs
    std::vector<LoonLobColumnConfig> lob_configs;
    if (lob_field_ids) {
      jsize n = env->GetArrayLength(lob_field_ids);
      jlong* fids = env->GetLongArrayElements(lob_field_ids, nullptr);
      jlong* inlines = lob_inline_thresholds ? env->GetLongArrayElements(lob_inline_thresholds, nullptr) : nullptr;
      jlong* maxfiles = lob_max_file_bytes ? env->GetLongArrayElements(lob_max_file_bytes, nullptr) : nullptr;
      jlong* flushes = lob_flush_thresholds ? env->GetLongArrayElements(lob_flush_thresholds, nullptr) : nullptr;

      lob_configs.resize(n);
      for (jsize i = 0; i < n; i++) {
        lob_configs[i].field_id = fids[i];
        auto path = (jstring)env->GetObjectArrayElement(lob_base_paths, i);
        lob_configs[i].lob_base_path = env->GetStringUTFChars(path, nullptr);
        lob_configs[i].inline_threshold = inlines ? inlines[i] : 256;
        lob_configs[i].max_lob_file_bytes = maxfiles ? maxfiles[i] : 64 * 1024 * 1024;
        lob_configs[i].flush_threshold_bytes = flushes ? flushes[i] : 16 * 1024 * 1024;
        lob_configs[i].rewrite_mode = false;
      }

      env->ReleaseLongArrayElements(lob_field_ids, fids, JNI_ABORT);
      if (inlines)
        env->ReleaseLongArrayElements(lob_inline_thresholds, inlines, JNI_ABORT);
      if (maxfiles)
        env->ReleaseLongArrayElements(lob_max_file_bytes, maxfiles, JNI_ABORT);
      if (flushes)
        env->ReleaseLongArrayElements(lob_flush_thresholds, flushes, JNI_ABORT);
    }

    LoonSegmentReaderConfig config;
    config.lob_columns = lob_configs.empty() ? nullptr : lob_configs.data();
    config.num_lob_columns = lob_configs.size();
    config.read_buffer_size = 0;  // use default

    LoonSegmentReaderHandle handle;
    LoonFFIResult result = loon_segment_reader_open(seg_path, version, schema, columns,
                                                    static_cast<int64_t>(num_columns), &config, properties, &handle);

    // Cleanup
    env->ReleaseStringUTFChars(segment_path, seg_path);
    FreeStringArray(env, columns, num_columns);
    for (jsize i = 0; i < static_cast<jsize>(lob_configs.size()); i++) {
      auto path = (jstring)env->GetObjectArrayElement(lob_base_paths, i);
      env->ReleaseStringUTFChars(path, lob_configs[i].lob_base_path);
    }

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return static_cast<jlong>(handle);
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetStream(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong handle) {
  try {
    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(calloc(1, sizeof(ArrowArrayStream)));
    LoonFFIResult result = loon_segment_reader_get_stream(static_cast<LoonSegmentReaderHandle>(handle), stream);

    if (!loon_ffi_is_success(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return reinterpret_cast<jlong>(stream);
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderTake(
    JNIEnv* env, jobject obj, jlong handle, jlongArray row_indices, jint parallelism) {
  try {
    if (!row_indices) {
      env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "row_indices must not be null");
      return -1;
    }

    jsize n = env->GetArrayLength(row_indices);
    jlong* indices = env->GetLongArrayElements(row_indices, nullptr);

    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(calloc(1, sizeof(ArrowArrayStream)));
    LoonFFIResult result = loon_segment_reader_take(static_cast<LoonSegmentReaderHandle>(handle),
                                                    reinterpret_cast<const int64_t*>(indices), static_cast<int64_t>(n),
                                                    static_cast<int64_t>(parallelism > 0 ? parallelism : 1), stream);

    env->ReleaseLongArrayElements(row_indices, indices, JNI_ABORT);

    if (!loon_ffi_is_success(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return reinterpret_cast<jlong>(stream);
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetFilteredStream(JNIEnv* env,
                                                                                                  jobject obj,
                                                                                                  jlong handle,
                                                                                                  jstring predicate) {
  try {
    const char* pred = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;

    ArrowArrayStream* stream = static_cast<ArrowArrayStream*>(calloc(1, sizeof(ArrowArrayStream)));
    LoonFFIResult result =
        loon_segment_reader_get_filtered_stream(static_cast<LoonSegmentReaderHandle>(handle), pred, stream);

    if (pred) {
      env->ReleaseStringUTFChars(predicate, pred);
    }

    if (!loon_ffi_is_success(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return reinterpret_cast<jlong>(stream);
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetChunkReader(
    JNIEnv* env, jobject obj, jlong handle, jlong column_group_index, jobjectArray needed_columns) {
  try {
    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);

    LoonChunkReaderHandle chunk_handle;
    LoonFFIResult result = loon_segment_reader_get_chunk_reader(static_cast<LoonSegmentReaderHandle>(handle),
                                                                static_cast<int64_t>(column_group_index), columns,
                                                                num_columns, &chunk_handle);

    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return static_cast<jlong>(chunk_handle);
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderDestroy(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jlong handle) {
  loon_segment_reader_destroy(static_cast<LoonSegmentReaderHandle>(handle));
}
