// Copyright 2026 Zilliz
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

// V2 (non-manifest) LoonColumnGroups construction from Scala-provided arrays.
//
// Used by the spark-connector's V2 read path when the caller already knows
// the segment's column-group layout (recovered from the snapshot AVRO +
// parquet footer kv-metadata) and does NOT want to resolve a milvus-storage
// `.milvus_manifest` file.
//
// Allocation lives in `src/ffi/v2_column_groups_builder.cpp` so it is part
// of libmilvus-storage and can be covered by the gtest suite. This file
// only adapts JNI inputs/outputs.

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_internal/v2_column_groups_builder.h"
#include "milvus-storage/ffi_jni.h"

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void ThrowJava(JNIEnv* env, const char* cls_name, const std::string& msg) {
  jclass exc = env->FindClass(cls_name);
  if (exc != nullptr) {
    env->ThrowNew(exc, msg.c_str());
  }
}

// Copy a jstring into std::string; returns false if the jstring is null.
bool ReadJString(JNIEnv* env, jstring js, std::string* out) {
  if (js == nullptr) return false;
  const char* s = env->GetStringUTFChars(js, nullptr);
  if (s == nullptr) return false;
  out->assign(s);
  env->ReleaseStringUTFChars(js, s);
  return true;
}

}  // namespace

// Other *_jni.cpp files pick up C linkage via declarations already wrapped in
// `extern "C" { ... }` inside ffi_jni.h. These entry points are new and have
// no matching declaration there, so wrap the definitions explicitly to keep
// the symbol name unmangled — otherwise JNI's runtime lookup fails with
// UnsatisfiedLinkError.
extern "C" {

// -----------------------------------------------------------------------------
//  io.milvus.storage.MilvusStorageColumnGroupsNative.createFromGroups
// -----------------------------------------------------------------------------
//  Java signature:
//    long createFromGroups(
//        String[][] columnsPerGroup,
//        String[][] filesPerGroup,
//        long[][] fileRowCountsPerGroup)
//
//  Returns a LoonColumnGroups* as jlong; caller frees via destroy().
//  Throws IllegalArgumentException / RuntimeException on any error.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_createFromGroups(
    JNIEnv* env,
    jobject /*obj*/,
    jobjectArray columns_per_group,
    jobjectArray files_per_group,
    jobjectArray file_row_counts_per_group) {
  try {
    if (!columns_per_group || !files_per_group || !file_row_counts_per_group) {
      ThrowJava(env, "java/lang/IllegalArgumentException",
                "columnsPerGroup / filesPerGroup / fileRowCountsPerGroup must not be null");
      return 0;
    }

    const jsize num_groups = env->GetArrayLength(columns_per_group);
    std::vector<std::vector<std::string>> cols(num_groups);
    std::vector<std::vector<std::string>> files(num_groups);
    std::vector<std::vector<int64_t>> row_counts(num_groups);

    // Surface per-group null inputs from Java before handing off to the
    // builder — the builder validates sizes, not nullity.
    for (jsize g = 0; g < num_groups; ++g) {
      jobjectArray jcols = static_cast<jobjectArray>(env->GetObjectArrayElement(columns_per_group, g));
      jobjectArray jfiles = static_cast<jobjectArray>(env->GetObjectArrayElement(files_per_group, g));
      jlongArray jrcs = static_cast<jlongArray>(env->GetObjectArrayElement(file_row_counts_per_group, g));
      if (!jcols || !jfiles || !jrcs) {
        ThrowJava(env, "java/lang/IllegalArgumentException",
                  "group[" + std::to_string(g) + "]: columns/files/rowCounts must not be null");
        return 0;
      }

      const jsize num_cols = env->GetArrayLength(jcols);
      cols[g].resize(num_cols);
      for (jsize c = 0; c < num_cols; ++c) {
        jstring js = static_cast<jstring>(env->GetObjectArrayElement(jcols, c));
        if (!ReadJString(env, js, &cols[g][c])) {
          ThrowJava(env, "java/lang/IllegalArgumentException",
                    "group[" + std::to_string(g) + "].columns[" + std::to_string(c) + "] is null");
          return 0;
        }
        env->DeleteLocalRef(js);
      }

      const jsize num_files = env->GetArrayLength(jfiles);
      files[g].resize(num_files);
      for (jsize f = 0; f < num_files; ++f) {
        jstring js = static_cast<jstring>(env->GetObjectArrayElement(jfiles, f));
        if (!ReadJString(env, js, &files[g][f])) {
          ThrowJava(env, "java/lang/IllegalArgumentException",
                    "group[" + std::to_string(g) + "].files[" + std::to_string(f) + "] is null");
          return 0;
        }
        env->DeleteLocalRef(js);
      }

      const jsize num_rcs = env->GetArrayLength(jrcs);
      row_counts[g].resize(num_rcs);
      env->GetLongArrayRegion(jrcs, 0, num_rcs, reinterpret_cast<jlong*>(row_counts[g].data()));

      env->DeleteLocalRef(jcols);
      env->DeleteLocalRef(jfiles);
      env->DeleteLocalRef(jrcs);
    }

    LoonColumnGroups* cgroups = milvus_storage::BuildLoonColumnGroups(cols, files, row_counts);
    return reinterpret_cast<jlong>(cgroups);
  } catch (const std::invalid_argument& e) {
    ThrowJava(env, "java/lang/IllegalArgumentException", e.what());
    return 0;
  } catch (const std::exception& e) {
    ThrowJava(env, "java/lang/RuntimeException", std::string("createFromGroups failed: ") + e.what());
    return 0;
  }
}

// -----------------------------------------------------------------------------
//  io.milvus.storage.MilvusStorageColumnGroupsNative.destroy(long)
// -----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_destroy(
    JNIEnv* /*env*/, jobject /*obj*/, jlong ptr) {
  loon_column_groups_destroy(reinterpret_cast<LoonColumnGroups*>(ptr));
}

}  // extern "C"
