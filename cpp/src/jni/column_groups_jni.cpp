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

// Column-group metadata shared by writer results and manifest-backed readers.
// Manifest column groups are borrowed; the manifest retains ownership.

#include "jni_raii.h"
#include <limits>
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

extern "C" {
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
