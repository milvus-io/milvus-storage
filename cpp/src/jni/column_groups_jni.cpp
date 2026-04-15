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

// Direct construction of LoonColumnGroups from Scala-provided arrays.
//
// Used by the spark-connector's V2 read path when the caller already knows
// the segment's column-group layout (recovered from the snapshot AVRO +
// parquet footer kv-metadata) and does NOT want to resolve a milvus-storage
// `.milvus_manifest` file.
//
// Allocations here match the `delete[]` expectations of
// `loon_column_groups_destroy` in ffi/column_groups_c.cpp:
//   - every string is allocated with `new char[]` + std::strcpy
//   - array-of-pointers and array-of-structs use `new T[]`
//   - the outer LoonColumnGroups is `new LoonColumnGroups`.

#include "milvus-storage/ffi_c.h"
#include "milvus-storage/ffi_jni.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

// Allocate a null-terminated copy of `s` that `delete[]` can reclaim.
char* dup_cstr(const char* s) {
  const size_t n = std::strlen(s);
  char* p = new char[n + 1];
  std::memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

// Release a partially-constructed LoonColumnGroups on error paths.
void destroy_partial(LoonColumnGroups* cg) {
  if (!cg) return;
  loon_column_groups_destroy(cg);
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
//  start_index / end_index per file are computed as cumulative row offsets
//  within each column group: file 0 covers rows [0, n0), file 1 covers
//  [n0, n0+n1), etc. The packed reader rejects negative end_index, so we
//  cannot pass `-1` as a "whole file" sentinel — the AVRO's
//  `AvroBinlog.entries_num` provides the per-file row counts.
//
//  Returns a LoonColumnGroups* as jlong; caller frees via destroy().
//  Throws RuntimeException on any error.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_createFromGroups(
    JNIEnv* env,
    jobject /*obj*/,
    jobjectArray columns_per_group,
    jobjectArray files_per_group,
    jobjectArray file_row_counts_per_group) {
  LoonColumnGroups* cgroups = nullptr;
  try {
    if (!columns_per_group || !files_per_group || !file_row_counts_per_group) {
      jclass exc = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exc, "columnsPerGroup / filesPerGroup / fileRowCountsPerGroup must not be null");
      return 0;
    }
    const jsize num_groups_cols = env->GetArrayLength(columns_per_group);
    const jsize num_groups_files = env->GetArrayLength(files_per_group);
    const jsize num_groups_rc = env->GetArrayLength(file_row_counts_per_group);
    if (num_groups_cols != num_groups_files || num_groups_cols != num_groups_rc) {
      jclass exc = env->FindClass("java/lang/IllegalArgumentException");
      std::string msg = "per-group array length mismatch: cols=" + std::to_string(num_groups_cols) +
                        ", files=" + std::to_string(num_groups_files) +
                        ", rowCounts=" + std::to_string(num_groups_rc);
      env->ThrowNew(exc, msg.c_str());
      return 0;
    }
    if (num_groups_cols == 0) {
      jclass exc = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exc, "at least one column group is required");
      return 0;
    }

    cgroups = new LoonColumnGroups{};
    cgroups->num_of_column_groups = static_cast<uint32_t>(num_groups_cols);
    cgroups->column_group_array = new LoonColumnGroup[num_groups_cols]{};

    for (jsize g = 0; g < num_groups_cols; ++g) {
      jobjectArray cols =
          static_cast<jobjectArray>(env->GetObjectArrayElement(columns_per_group, g));
      jobjectArray files =
          static_cast<jobjectArray>(env->GetObjectArrayElement(files_per_group, g));
      jlongArray rcs =
          static_cast<jlongArray>(env->GetObjectArrayElement(file_row_counts_per_group, g));
      if (!cols || !files || !rcs) {
        destroy_partial(cgroups);
        jclass exc = env->FindClass("java/lang/IllegalArgumentException");
        std::string msg = "group[" + std::to_string(g) + "]: columns/files/rowCounts must not be null";
        env->ThrowNew(exc, msg.c_str());
        return 0;
      }

      const jsize num_cols = env->GetArrayLength(cols);
      const jsize num_files = env->GetArrayLength(files);
      const jsize num_rcs = env->GetArrayLength(rcs);
      if (num_cols == 0 || num_files == 0) {
        destroy_partial(cgroups);
        jclass exc = env->FindClass("java/lang/IllegalArgumentException");
        std::string msg =
            "group[" + std::to_string(g) + "]: columns/files must be non-empty";
        env->ThrowNew(exc, msg.c_str());
        return 0;
      }
      if (num_rcs != num_files) {
        destroy_partial(cgroups);
        jclass exc = env->FindClass("java/lang/IllegalArgumentException");
        std::string msg = "group[" + std::to_string(g) + "]: rowCounts.length (" +
                          std::to_string(num_rcs) + ") != files.length (" +
                          std::to_string(num_files) + ")";
        env->ThrowNew(exc, msg.c_str());
        return 0;
      }

      LoonColumnGroup& out = cgroups->column_group_array[g];
      out.num_of_columns = static_cast<uint32_t>(num_cols);
      out.columns = new const char*[num_cols]{};
      out.format = dup_cstr("parquet");
      out.num_of_files = static_cast<uint32_t>(num_files);
      out.files = new LoonColumnGroupFile[num_files]{};

      // Copy column names (field-id-as-string).
      for (jsize c = 0; c < num_cols; ++c) {
        jstring js = static_cast<jstring>(env->GetObjectArrayElement(cols, c));
        if (!js) {
          destroy_partial(cgroups);
          jclass exc = env->FindClass("java/lang/IllegalArgumentException");
          std::string msg = "group[" + std::to_string(g) + "].columns[" +
                            std::to_string(c) + "] is null";
          env->ThrowNew(exc, msg.c_str());
          return 0;
        }
        const char* s = env->GetStringUTFChars(js, nullptr);
        out.columns[c] = dup_cstr(s);
        env->ReleaseStringUTFChars(js, s);
        env->DeleteLocalRef(js);
      }

      // Pull row counts in a single bulk copy.
      std::vector<int64_t> row_counts(num_files);
      env->GetLongArrayRegion(rcs, 0, num_files, reinterpret_cast<jlong*>(row_counts.data()));

      // Cumulative row offsets within the column group: file i covers
      // [start_i, end_i) where start_i = sum(rowCounts[0..i)) and
      // end_i = start_i + rowCounts[i].
      int64_t cumulative = 0;
      for (jsize f = 0; f < num_files; ++f) {
        jstring jp = static_cast<jstring>(env->GetObjectArrayElement(files, f));
        if (!jp) {
          destroy_partial(cgroups);
          jclass exc = env->FindClass("java/lang/IllegalArgumentException");
          std::string msg = "group[" + std::to_string(g) + "].files[" +
                            std::to_string(f) + "] is null";
          env->ThrowNew(exc, msg.c_str());
          return 0;
        }
        const char* s = env->GetStringUTFChars(jp, nullptr);
        out.files[f].path = dup_cstr(s);
        env->ReleaseStringUTFChars(jp, s);
        env->DeleteLocalRef(jp);
        const int64_t n = row_counts[f];
        out.files[f].start_index = cumulative;
        out.files[f].end_index = cumulative + n;
        cumulative += n;
        out.files[f].property_keys = nullptr;
        out.files[f].property_values = nullptr;
        out.files[f].num_properties = 0;
      }

      env->DeleteLocalRef(cols);
      env->DeleteLocalRef(files);
      env->DeleteLocalRef(rcs);
    }

    return reinterpret_cast<jlong>(cgroups);
  } catch (const std::exception& e) {
    destroy_partial(cgroups);
    jclass exc = env->FindClass("java/lang/RuntimeException");
    std::string msg = std::string("createFromGroups failed: ") + e.what();
    env->ThrowNew(exc, msg.c_str());
    return 0;
  }
}

// -----------------------------------------------------------------------------
//  io.milvus.storage.MilvusStorageColumnGroupsNative.destroy(long)
// -----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_destroy(
    JNIEnv* /*env*/, jobject /*obj*/, jlong ptr) {
  if (ptr == 0) return;
  LoonColumnGroups* cg = reinterpret_cast<LoonColumnGroups*>(ptr);
  loon_column_groups_destroy(cg);
}

}  // extern "C"
