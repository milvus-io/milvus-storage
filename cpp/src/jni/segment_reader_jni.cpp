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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ==================== JNI SegmentReader Implementation ====================

namespace {

bool CheckArrayLength(JNIEnv* env, jarray array, jsize expected, const char* name, bool required) {
  if (array == nullptr) {
    if (required) {
      char message[160];
      std::snprintf(message, sizeof(message), "%s must not be null when LOB field IDs are present", name);
      ThrowJavaException(env, "java/lang/IllegalArgumentException", message);
      return false;
    }
    return true;
  }

  const jsize actual = env->GetArrayLength(array);
  if (env->ExceptionCheck()) {
    return false;
  }
  if (actual != expected) {
    char message[160];
    std::snprintf(message, sizeof(message), "%s length (%d) must equal lobFieldIds length (%d)", name, actual,
                  expected);
    ThrowJavaException(env, "java/lang/IllegalArgumentException", message);
    return false;
  }
  return true;
}

bool CopyLongArray(JNIEnv* env, jlongArray array, jsize expected, int64_t default_value, std::vector<int64_t>* output) {
  output->assign(static_cast<size_t>(expected), default_value);
  if (array == nullptr || expected == 0) {
    return true;
  }
  std::vector<jlong> java_values(static_cast<size_t>(expected));
  env->GetLongArrayRegion(array, 0, expected, java_values.data());
  if (env->ExceptionCheck()) {
    return false;
  }
  for (jsize i = 0; i < expected; ++i) {
    (*output)[static_cast<size_t>(i)] = static_cast<int64_t>(java_values[static_cast<size_t>(i)]);
  }
  return true;
}

ArrowArrayStream* AllocateStream(JNIEnv* env) {
  auto* stream = static_cast<ArrowArrayStream*>(calloc(1, sizeof(ArrowArrayStream)));
  if (stream == nullptr) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Unexpected native allocation failure for ArrowArrayStream");
  }
  return stream;
}

}  // namespace

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
    if (segment_path == nullptr) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "segmentPath must not be null");
      return -1;
    }

    auto* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    auto* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    if (lob_field_ids == nullptr) {
      if (lob_base_paths != nullptr || lob_inline_thresholds != nullptr || lob_max_file_bytes != nullptr ||
          lob_flush_thresholds != nullptr) {
        ThrowJavaException(env, "java/lang/IllegalArgumentException",
                           "LOB configuration arrays must all be null when lobFieldIds is null");
        return -1;
      }
    }

    const jsize num_lob_columns = lob_field_ids == nullptr ? 0 : env->GetArrayLength(lob_field_ids);
    if (env->ExceptionCheck()) {
      return -1;
    }
    if (!CheckArrayLength(env, lob_base_paths, num_lob_columns, "lobBasePaths", num_lob_columns > 0) ||
        !CheckArrayLength(env, lob_inline_thresholds, num_lob_columns, "lobInlineThresholds", false) ||
        !CheckArrayLength(env, lob_max_file_bytes, num_lob_columns, "lobMaxFileBytes", false) ||
        !CheckArrayLength(env, lob_flush_thresholds, num_lob_columns, "lobFlushThresholds", false)) {
      return -1;
    }

    std::vector<int64_t> field_ids;
    std::vector<int64_t> inline_thresholds;
    std::vector<int64_t> max_file_bytes;
    std::vector<int64_t> flush_thresholds;
    if (!CopyLongArray(env, lob_field_ids, num_lob_columns, 0, &field_ids) ||
        !CopyLongArray(env, lob_inline_thresholds, num_lob_columns, 256, &inline_thresholds) ||
        !CopyLongArray(env, lob_max_file_bytes, num_lob_columns, 64 * 1024 * 1024, &max_file_bytes) ||
        !CopyLongArray(env, lob_flush_thresholds, num_lob_columns, 16 * 1024 * 1024, &flush_thresholds)) {
      return -1;
    }

    std::vector<LoonLobColumnConfig> lob_configs(static_cast<size_t>(num_lob_columns));

    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck()) {
      return -1;
    }

    size_t num_lob_paths = 0;
    const char** lob_paths = ConvertFromJavaStringArray(env, lob_base_paths, &num_lob_paths);
    if (env->ExceptionCheck()) {
      FreeStringArray(env, columns, num_columns);
      return -1;
    }
    if (num_lob_paths != static_cast<size_t>(num_lob_columns)) {
      FreeStringArray(env, lob_paths, num_lob_paths);
      FreeStringArray(env, columns, num_columns);
      ThrowJavaException(env, "java/lang/IllegalArgumentException",
                         "lobBasePaths length must equal lobFieldIds length");
      return -1;
    }

    for (jsize i = 0; i < num_lob_columns; ++i) {
      auto& config = lob_configs[static_cast<size_t>(i)];
      config.field_id = field_ids[static_cast<size_t>(i)];
      config.lob_base_path = lob_paths[static_cast<size_t>(i)];
      config.inline_threshold = inline_thresholds[static_cast<size_t>(i)];
      config.max_lob_file_bytes = max_file_bytes[static_cast<size_t>(i)];
      config.flush_threshold_bytes = flush_thresholds[static_cast<size_t>(i)];
      config.rewrite_mode = false;
    }

    const char* seg_path = env->GetStringUTFChars(segment_path, nullptr);
    if (seg_path == nullptr) {
      FreeStringArray(env, lob_paths, num_lob_paths);
      FreeStringArray(env, columns, num_columns);
      return -1;
    }

    LoonSegmentReaderConfig config{};
    config.lob_columns = lob_configs.empty() ? nullptr : lob_configs.data();
    config.num_lob_columns = lob_configs.size();
    config.read_buffer_size = 0;  // use default

    LoonSegmentReaderHandle handle = 0;
    LoonFFIResult result = loon_segment_reader_open(seg_path, version, schema, columns,
                                                    static_cast<int64_t>(num_columns), &config, properties, &handle);

    env->ReleaseStringUTFChars(segment_path, seg_path);
    FreeStringArray(env, lob_paths, num_lob_paths);
    FreeStringArray(env, columns, num_columns);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return static_cast<jlong>(handle);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetStream(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong handle) {
  try {
    ArrowArrayStream* stream = AllocateStream(env);
    if (stream == nullptr) {
      return -1;
    }
    LoonFFIResult result = loon_segment_reader_get_stream(static_cast<LoonSegmentReaderHandle>(handle), stream);

    if (!loon_ffi_is_success(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return reinterpret_cast<jlong>(stream);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderTake(
    JNIEnv* env, jobject obj, jlong handle, jlongArray row_indices, jint parallelism) {
  try {
    if (!row_indices) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "rowIndices must not be null");
      return -1;
    }
    if (parallelism < 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "parallelism must be greater than or equal to 0");
      return -1;
    }

    jsize n = env->GetArrayLength(row_indices);
    if (env->ExceptionCheck()) {
      return -1;
    }
    if (n == 0) {
      ThrowJavaException(env, "java/lang/IllegalArgumentException", "rowIndices must not be empty");
      return -1;
    }
    std::vector<jlong> java_indices(static_cast<size_t>(n));
    env->GetLongArrayRegion(row_indices, 0, n, java_indices.data());
    if (env->ExceptionCheck()) {
      return -1;
    }
    std::vector<int64_t> indices(static_cast<size_t>(n));
    for (jsize i = 0; i < n; ++i) {
      indices[static_cast<size_t>(i)] = static_cast<int64_t>(java_indices[static_cast<size_t>(i)]);
    }

    ArrowArrayStream* stream = AllocateStream(env);
    if (stream == nullptr) {
      return -1;
    }
    LoonFFIResult result = loon_segment_reader_take(static_cast<LoonSegmentReaderHandle>(handle), indices.data(),
                                                    static_cast<int64_t>(n), static_cast<int64_t>(parallelism), stream);

    if (!loon_ffi_is_success(&result)) {
      free(stream);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return reinterpret_cast<jlong>(stream);
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetFilteredStream(JNIEnv* env,
                                                                                                  jobject obj,
                                                                                                  jlong handle,
                                                                                                  jstring predicate) {
  try {
    ArrowArrayStream* stream = AllocateStream(env);
    if (stream == nullptr) {
      return -1;
    }
    const char* pred = predicate ? env->GetStringUTFChars(predicate, nullptr) : nullptr;
    if (predicate != nullptr && pred == nullptr) {
      free(stream);
      return -1;
    }
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
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderGetChunkReader(
    JNIEnv* env, jobject obj, jlong handle, jlong column_group_index, jobjectArray needed_columns) {
  try {
    size_t num_columns = 0;
    const char** columns = ConvertFromJavaStringArray(env, needed_columns, &num_columns);
    if (env->ExceptionCheck()) {
      return -1;
    }

    LoonChunkReaderHandle chunk_handle = 0;
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
  } catch (...) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native operation failed");
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentReader_segmentReaderDestroy(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jlong handle) {
  loon_segment_reader_destroy(static_cast<LoonSegmentReaderHandle>(handle));
}
