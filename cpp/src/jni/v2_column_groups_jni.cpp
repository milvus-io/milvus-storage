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

#include "milvus-storage/ffi_internal/v2_column_groups_builder.h"
#include "jni_raii.h"
#include <limits>
#include <memory>
#include <vector>

using namespace milvus_storage::jni;

namespace {
const LoonColumnGroup* Group(JNIEnv* env, jlong pointer, jint index) {
  if (pointer == 0) {
    Throw(env, "java/lang/IllegalArgumentException", "Column groups must not be null");
    return nullptr;
  }
  const auto* groups = reinterpret_cast<const LoonColumnGroups*>(pointer);
  if (index < 0 || static_cast<uint32_t>(index) >= groups->num_of_column_groups) {
    Throw(env, "java/lang/IndexOutOfBoundsException", "Column group index is out of bounds");
    return nullptr;
  }
  return &groups->column_group_array[index];
}

jobjectArray Strings(JNIEnv* env, const char* const* values, uint32_t count) {
  if (count > static_cast<uint32_t>(std::numeric_limits<jsize>::max())) {
    Throw(env, "java/lang/IllegalArgumentException", "Column group metadata exceeds the JVM array limit");
    return nullptr;
  }
  LocalRef<jclass> cls(env, env->FindClass("java/lang/String"));
  if (cls.get() == nullptr)
    return nullptr;
  LocalRef<jobjectArray> result(env, env->NewObjectArray(static_cast<jsize>(count), cls.get(), nullptr));
  if (result.get() == nullptr)
    return nullptr;
  for (uint32_t i = 0; i < count; ++i) {
    LocalRef<jstring> value(env, env->NewStringUTF(values[i] == nullptr ? "" : values[i]));
    if (value.get() == nullptr)
      return nullptr;
    env->SetObjectArrayElement(result.get(), static_cast<jsize>(i), value.get());
    if (env->ExceptionCheck())
      return nullptr;
  }
  return result.release();
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
//        long[][] fileRowCountsPerGroup,
//        String format)
//
//  Returns a LoonColumnGroups* as jlong; caller frees via destroy().
//  Throws IllegalArgumentException / RuntimeException on any error.
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_createFromGroups(
    JNIEnv* env, jobject, jobjectArray columns, jobjectArray files, jobjectArray row_counts, jstring format) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 format_name(env, format);
    if (!format_name.valid())
      return jlong{0};
    if (columns == nullptr || files == nullptr || row_counts == nullptr) {
      Throw(env, "java/lang/IllegalArgumentException", "Column group arrays must not be null");
      return jlong{0};
    }
    const jsize count = env->GetArrayLength(columns);
    if (env->GetArrayLength(files) != count || env->GetArrayLength(row_counts) != count) {
      Throw(env, "java/lang/IllegalArgumentException", "Column group array lengths must match");
      return jlong{0};
    }
    std::vector<std::vector<std::string>> column_values(count), file_values(count);
    std::vector<std::vector<int64_t>> rows(count);
    for (jsize i = 0; i < count; ++i) {
      LocalRef<jobjectArray> group_columns(env, static_cast<jobjectArray>(env->GetObjectArrayElement(columns, i)));
      if (env->ExceptionCheck())
        return jlong{0};
      if (!ReadStrings(env, group_columns.get(), &column_values[i]))
        return jlong{0};
      LocalRef<jobjectArray> group_files(env, static_cast<jobjectArray>(env->GetObjectArrayElement(files, i)));
      if (env->ExceptionCheck())
        return jlong{0};
      if (!ReadStrings(env, group_files.get(), &file_values[i]))
        return jlong{0};
      LocalRef<jlongArray> group_rows(env, static_cast<jlongArray>(env->GetObjectArrayElement(row_counts, i)));
      if (env->ExceptionCheck())
        return jlong{0};
      if (group_rows.get() == nullptr) {
        Throw(env, "java/lang/IllegalArgumentException", "File row counts must not be null");
        return jlong{0};
      }
      const jsize size = env->GetArrayLength(group_rows.get());
      rows[i].resize(size);
      if (size != 0)
        env->GetLongArrayRegion(group_rows.get(), 0, size, reinterpret_cast<jlong*>(rows[i].data()));
      if (env->ExceptionCheck())
        return jlong{0};
    }
    return reinterpret_cast<jlong>(
        milvus_storage::BuildLoonColumnGroups(column_values, file_values, rows, format_name.get()));
  });
}

// -----------------------------------------------------------------------------
//  io.milvus.storage.MilvusStorageColumnGroupsNative.destroy(long)
// -----------------------------------------------------------------------------
JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_destroy(JNIEnv* env,
                                                                                      jobject,
                                                                                      jlong pointer) {
  GuardVoid(env, [&] { loon_column_groups_destroy(reinterpret_cast<LoonColumnGroups*>(pointer)); });
}

JNIEXPORT jint JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_count(JNIEnv* env,
                                                                                    jobject,
                                                                                    jlong pointer) {
  if (pointer == 0) {
    Throw(env, "java/lang/IllegalArgumentException", "Column groups must not be null");
    return 0;
  }
  const auto count = reinterpret_cast<const LoonColumnGroups*>(pointer)->num_of_column_groups;
  if (count > static_cast<uint32_t>(std::numeric_limits<jint>::max())) {
    Throw(env, "java/lang/ArithmeticException", "Too many column groups");
    return 0;
  }
  return static_cast<jint>(count);
}

JNIEXPORT jobjectArray JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_columns(JNIEnv* env,
                                                                                              jobject,
                                                                                              jlong pointer,
                                                                                              jint index) {
  return Guard<jobjectArray>(env, nullptr, [&]() -> jobjectArray {
    const auto* group = Group(env, pointer, index);
    return group == nullptr ? nullptr : Strings(env, group->columns, group->num_of_columns);
  });
}

JNIEXPORT jobjectArray JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_files(JNIEnv* env,
                                                                                            jobject,
                                                                                            jlong pointer,
                                                                                            jint index) {
  return Guard<jobjectArray>(env, nullptr, [&]() -> jobjectArray {
    const auto* group = Group(env, pointer, index);
    if (group == nullptr)
      return nullptr;
    std::vector<const char*> paths;
    paths.reserve(group->num_of_files);
    for (uint32_t i = 0; i < group->num_of_files; ++i) paths.push_back(group->files[i].path);
    return Strings(env, paths.data(), group->num_of_files);
  });
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_fileRowCounts(JNIEnv* env,
                                                                                                  jobject,
                                                                                                  jlong pointer,
                                                                                                  jint index) {
  return Guard<jlongArray>(env, nullptr, [&]() -> jlongArray {
    const auto* group = Group(env, pointer, index);
    if (group == nullptr)
      return nullptr;
    if (group->num_of_files > static_cast<uint32_t>(std::numeric_limits<jsize>::max())) {
      Throw(env, "java/lang/IllegalArgumentException", "Too many column group files");
      return nullptr;
    }
    std::vector<jlong> rows;
    rows.reserve(group->num_of_files);
    for (uint32_t i = 0; i < group->num_of_files; ++i) {
      const auto& file = group->files[i];
      if (file.start_index < 0 || file.end_index < file.start_index) {
        Throw(env, "java/lang/IllegalStateException", "Invalid column group file row range");
        return nullptr;
      }
      rows.push_back(file.end_index - file.start_index);
    }
    jlongArray result = env->NewLongArray(static_cast<jsize>(rows.size()));
    if (result == nullptr)
      return nullptr;
    if (!rows.empty())
      env->SetLongArrayRegion(result, 0, static_cast<jsize>(rows.size()), rows.data());
    return env->ExceptionCheck() ? nullptr : result;
  });
}

JNIEXPORT jstring JNICALL Java_io_milvus_storage_MilvusStorageColumnGroupsNative_format(JNIEnv* env,
                                                                                        jobject,
                                                                                        jlong pointer,
                                                                                        jint index) {
  const auto* group = Group(env, pointer, index);
  return group == nullptr ? nullptr : env->NewStringUTF(group->format == nullptr ? "" : group->format);
}
}  // extern "C"
