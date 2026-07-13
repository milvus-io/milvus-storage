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
#include <cassert>
#include <cstring>
#include <cstdlib>

// ==================== JNI Utility Functions ====================

/**
  Should use try-catch block to catch exceptions in C++ code.
  Then in the catch block, use JNI provided functions (such as ThrowNew) to throw a Java exception.
  Should return after throwing an exception, otherwise the function will continue to execute until the end of the
function.
**/
void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct LoonFFIResult* result) {
  if (loon_ffi_is_success(const_cast<LoonFFIResult*>(result))) {
    return;
  }

  const char* message = loon_ffi_get_errmsg(const_cast<LoonFFIResult*>(result));
  const char* exception_class = "java/lang/RuntimeException";

  if (result->err_code == loon_errcode_invalid_args) {
    exception_class = "java/lang/IllegalArgumentException";
  } else if (result->err_code == loon_errcode_memory) {
    exception_class = "java/lang/OutOfMemoryError";
  } else {
    exception_class = "java/lang/RuntimeException";
  }

  jclass exc_class = env->FindClass(exception_class);
  assert(exc_class != nullptr);
  env->ThrowNew(exc_class, message);
  return;
}

jobjectArray ConvertToJavaStringArray(JNIEnv* env, const char* const* strings, size_t count) {
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(count), string_class, nullptr);

  for (size_t i = 0; i < count; ++i) {
    jstring str = env->NewStringUTF(strings[i]);
    env->SetObjectArrayElement(result, static_cast<jsize>(i), str);
    env->DeleteLocalRef(str);
  }

  return result;
}

const char** ConvertFromJavaStringArray(JNIEnv* env, jobjectArray java_array, size_t* out_count) {
  if (java_array == nullptr) {
    *out_count = 0;
    return nullptr;
  }

  jsize length = env->GetArrayLength(java_array);
  *out_count = static_cast<size_t>(length);

  const char** strings = static_cast<const char**>(malloc(sizeof(char*) * length));
  for (jsize i = 0; i < length; ++i) {
    jstring jstr = static_cast<jstring>(env->GetObjectArrayElement(java_array, i));
    const char* str = env->GetStringUTFChars(jstr, nullptr);
    strings[i] = strdup(str);
    env->ReleaseStringUTFChars(jstr, str);
    env->DeleteLocalRef(jstr);
  }

  return strings;
}

void FreeStringArray(JNIEnv* env, const char** strings, size_t count) {
  if (strings != nullptr) {
    for (size_t i = 0; i < count; ++i) {
      free(const_cast<char*>(strings[i]));
    }
    free(strings);
  }
}
