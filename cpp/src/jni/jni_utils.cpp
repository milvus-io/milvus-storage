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
#include "jni_raii.h"
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>

using namespace milvus_storage::jni;

void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct LoonFFIResult* result) {
  if (env->ExceptionCheck() || loon_ffi_is_success(const_cast<LoonFFIResult*>(result)))
    return;
  LocalRef<jclass> cls(env, env->FindClass("io/milvus/storage/MilvusStorageException"));
  if (cls.get() == nullptr)
    return;
  jmethodID ctor = env->GetMethodID(cls.get(), "<init>", "(ILjava/lang/String;)V");
  if (ctor == nullptr)
    return;
  const char* message = loon_ffi_get_errmsg(const_cast<LoonFFIResult*>(result));
  LocalRef<jstring> text(env, env->NewStringUTF(message != nullptr ? message : ""));
  if (text.get() == nullptr)
    return;
  LocalRef<jobject> exception(env, env->NewObject(cls.get(), ctor, static_cast<jint>(result->err_code), text.get()));
  if (env->ExceptionCheck())
    return;
  if (exception.get() != nullptr)
    env->Throw(static_cast<jthrowable>(exception.get()));
}

jobjectArray ConvertToJavaStringArray(JNIEnv* env, const char* const* strings, size_t count) {
  return Guard<jobjectArray>(env, nullptr, [&]() -> jobjectArray {
    if (env->ExceptionCheck())
      return nullptr;
    if (count > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
      Throw(env, "java/lang/IllegalArgumentException", "String array exceeds Java array length limit");
      return nullptr;
    }
    LocalRef<jclass> string_class(env, env->FindClass("java/lang/String"));
    if (string_class.get() == nullptr)
      return nullptr;
    LocalRef<jobjectArray> result(env, env->NewObjectArray(static_cast<jsize>(count), string_class.get(), nullptr));
    if (result.get() == nullptr)
      return nullptr;
    for (size_t i = 0; i < count; ++i) {
      if (strings[i] == nullptr) {
        Throw(env, "java/lang/IllegalArgumentException", "String array element must not be null");
        return nullptr;
      }
      LocalRef<jstring> value(env, env->NewStringUTF(strings[i]));
      if (value.get() == nullptr)
        return nullptr;
      env->SetObjectArrayElement(result.get(), static_cast<jsize>(i), value.get());
      if (env->ExceptionCheck())
        return nullptr;
    }
    return result.release();
  });
}

const char** ConvertFromJavaStringArray(JNIEnv* env, jobjectArray java_array, size_t* out_count) {
  *out_count = 0;
  return Guard<const char**>(env, nullptr, [&]() -> const char** {
    if (env->ExceptionCheck() || java_array == nullptr)
      return nullptr;
    const jsize count = env->GetArrayLength(java_array);
    if (count == 0)
      return nullptr;
    auto cleanup = [env, count](const char** values) { FreeStringArray(env, values, static_cast<size_t>(count)); };
    std::unique_ptr<const char*, decltype(cleanup)> strings(
        static_cast<const char**>(calloc(static_cast<size_t>(count), sizeof(char*))), cleanup);
    if (strings.get() == nullptr) {
      Throw(env, "java/lang/OutOfMemoryError", "Cannot allocate native string array");
      return nullptr;
    }
    LocalRef<jclass> string_class(env, env->FindClass("java/lang/String"));
    if (string_class.get() == nullptr)
      return nullptr;
    for (jsize i = 0; i < count; ++i) {
      LocalRef<jobject> element(env, env->GetObjectArrayElement(java_array, i));
      if (env->ExceptionCheck())
        return nullptr;
      if (element.get() == nullptr || !env->IsInstanceOf(element.get(), string_class.get())) {
        Throw(env, "java/lang/IllegalArgumentException", "String array elements must be non-null strings");
        return nullptr;
      }
      ScopedUtf8 value(env, static_cast<jstring>(element.get()));
      if (!value.valid())
        return nullptr;
      strings.get()[i] = strdup(value.get());
      if (strings.get()[i] == nullptr) {
        Throw(env, "java/lang/OutOfMemoryError", "Cannot copy native string array element");
        return nullptr;
      }
    }
    *out_count = static_cast<size_t>(count);
    return strings.release();
  });
}

void FreeStringArray(JNIEnv*, const char** strings, size_t count) {
  if (strings != nullptr) {
    for (size_t i = 0; i < count; ++i) free(const_cast<char*>(strings[i]));
    free(strings);
  }
}
