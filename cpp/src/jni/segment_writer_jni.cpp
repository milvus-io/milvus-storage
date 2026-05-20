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
#include <cstring>
#include <string>
#include <vector>

// ==================== JNI SegmentWriter Implementation ====================

// Helper: build LoonLobColumnConfig array from parallel Java arrays.
// Caller must call FreeLobConfigs() after use.
static std::vector<LoonLobColumnConfig> BuildLobConfigs(JNIEnv* env,
                                                        jlongArray field_ids,
                                                        jobjectArray base_paths,
                                                        jlongArray inline_thresholds,
                                                        jlongArray max_file_bytes,
                                                        jlongArray flush_thresholds) {
  std::vector<LoonLobColumnConfig> configs;
  if (!field_ids)
    return configs;

  jsize n = env->GetArrayLength(field_ids);
  if (n == 0)
    return configs;

  jlong* fids = env->GetLongArrayElements(field_ids, nullptr);
  jlong* inlines = inline_thresholds ? env->GetLongArrayElements(inline_thresholds, nullptr) : nullptr;
  jlong* maxfiles = max_file_bytes ? env->GetLongArrayElements(max_file_bytes, nullptr) : nullptr;
  jlong* flushes = flush_thresholds ? env->GetLongArrayElements(flush_thresholds, nullptr) : nullptr;

  configs.resize(n);
  for (jsize i = 0; i < n; i++) {
    configs[i].field_id = fids[i];
    auto path = (jstring)env->GetObjectArrayElement(base_paths, i);
    configs[i].lob_base_path = env->GetStringUTFChars(path, nullptr);
    configs[i].inline_threshold = inlines ? inlines[i] : 256;
    configs[i].max_lob_file_bytes = maxfiles ? maxfiles[i] : 64 * 1024 * 1024;
    configs[i].flush_threshold_bytes = flushes ? flushes[i] : 16 * 1024 * 1024;
    configs[i].rewrite_mode = false;
  }

  env->ReleaseLongArrayElements(field_ids, fids, JNI_ABORT);
  if (inlines)
    env->ReleaseLongArrayElements(inline_thresholds, inlines, JNI_ABORT);
  if (maxfiles)
    env->ReleaseLongArrayElements(max_file_bytes, maxfiles, JNI_ABORT);
  if (flushes)
    env->ReleaseLongArrayElements(flush_thresholds, flushes, JNI_ABORT);

  return configs;
}

static void FreeLobConfigs(JNIEnv* env, std::vector<LoonLobColumnConfig>& configs, jobjectArray base_paths) {
  for (jsize i = 0; i < static_cast<jsize>(configs.size()); i++) {
    auto path = (jstring)env->GetObjectArrayElement(base_paths, i);
    env->ReleaseStringUTFChars(path, configs[i].lob_base_path);
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterNew(JNIEnv* env,
                                                                                    jobject obj,
                                                                                    jlong schema_ptr,
                                                                                    jstring segment_path,
                                                                                    jlongArray lob_field_ids,
                                                                                    jobjectArray lob_base_paths,
                                                                                    jlongArray lob_inline_thresholds,
                                                                                    jlongArray lob_max_file_bytes,
                                                                                    jlongArray lob_flush_thresholds,
                                                                                    jlong properties_ptr) {
  try {
    auto* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    auto* properties = reinterpret_cast<LoonProperties*>(properties_ptr);
    const char* seg_path = env->GetStringUTFChars(segment_path, nullptr);

    auto lob_configs = BuildLobConfigs(env, lob_field_ids, lob_base_paths, lob_inline_thresholds, lob_max_file_bytes,
                                       lob_flush_thresholds);

    LoonSegmentWriterConfig config;
    config.segment_path = seg_path;
    config.lob_columns = lob_configs.empty() ? nullptr : lob_configs.data();
    config.num_lob_columns = lob_configs.size();

    LoonSegmentWriterHandle handle;
    LoonFFIResult result = loon_segment_writer_new(schema, &config, properties, &handle);

    env->ReleaseStringUTFChars(segment_path, seg_path);
    if (!lob_configs.empty())
      FreeLobConfigs(env, lob_configs, lob_base_paths);

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

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterWrite(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong handle,
                                                                                     jlong array_ptr) {
  try {
    auto* array = reinterpret_cast<ArrowArray*>(array_ptr);
    LoonFFIResult result = loon_segment_writer_write(static_cast<LoonSegmentWriterHandle>(handle), array);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
    }
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterFlush(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong handle) {
  try {
    LoonFFIResult result = loon_segment_writer_flush(static_cast<LoonSegmentWriterHandle>(handle));
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
    }
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
}

// Close returns columnGroupsPtr. LOB file info is returned via output arrays.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterClose(JNIEnv* env,
                                                                                      jobject obj,
                                                                                      jlong handle,
                                                                                      jlongArray out_lob_field_ids,
                                                                                      jobjectArray out_lob_paths,
                                                                                      jlongArray out_lob_total_rows,
                                                                                      jlongArray out_lob_valid_rows,
                                                                                      jlongArray out_lob_file_sizes) {
  try {
    LoonSegmentWriteOutput output;
    memset(&output, 0, sizeof(output));
    LoonFFIResult result = loon_segment_writer_close(static_cast<LoonSegmentWriterHandle>(handle), &output);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    // Fill LOB file info into output arrays if provided and there are LOB files
    if (output.num_lob_files > 0 && out_lob_field_ids && out_lob_paths) {
      jsize n = static_cast<jsize>(output.num_lob_files);
      std::vector<jlong> field_ids(n), total_rows(n), valid_rows(n), file_sizes(n);
      for (jsize i = 0; i < n; i++) {
        field_ids[i] = output.lob_files[i].field_id;
        total_rows[i] = output.lob_files[i].total_rows;
        valid_rows[i] = output.lob_files[i].valid_rows;
        file_sizes[i] = output.lob_files[i].file_size_bytes;

        jstring path = env->NewStringUTF(output.lob_files[i].path);
        env->SetObjectArrayElement(out_lob_paths, i, path);
      }
      env->SetLongArrayRegion(out_lob_field_ids, 0, n, field_ids.data());
      if (out_lob_total_rows)
        env->SetLongArrayRegion(out_lob_total_rows, 0, n, total_rows.data());
      if (out_lob_valid_rows)
        env->SetLongArrayRegion(out_lob_valid_rows, 0, n, valid_rows.data());
      if (out_lob_file_sizes)
        env->SetLongArrayRegion(out_lob_file_sizes, 0, n, file_sizes.data());
    }

    jlong cg_ptr = reinterpret_cast<jlong>(output.column_groups);
    loon_segment_write_output_free(&output);
    return cg_ptr;
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusSegmentWriter_segmentWriterDestroy(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jlong handle) {
  loon_segment_writer_destroy(static_cast<LoonSegmentWriterHandle>(handle));
}
