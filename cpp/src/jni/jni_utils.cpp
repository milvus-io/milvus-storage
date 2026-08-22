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
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ==================== JNI Utility Functions ====================

/**
  Should use try-catch block to catch exceptions in C++ code.
  Then in the catch block, use JNI provided functions (such as ThrowNew) to throw a Java exception.
  Should return after throwing an exception, otherwise the function will continue to execute until the end of the
function.
**/
void ThrowJavaException(JNIEnv* env, const char* exception_class, const char* message) {
  if (env == nullptr || env->ExceptionCheck()) {
    return;
  }

  jclass exc_class = env->FindClass(exception_class);
  if (exc_class == nullptr) {
    // FindClass leaves its own Java exception pending. Never assert/abort while
    // trying to report the original failure (especially under memory pressure).
    return;
  }
  env->ThrowNew(exc_class, message != nullptr ? message : "Native operation failed");
  env->DeleteLocalRef(exc_class);
}

void ThrowJavaExceptionFromFFIResult(JNIEnv* env, const struct LoonFFIResult* result) {
  if (result == nullptr) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native result must not be null");
    return;
  }
  if (loon_ffi_is_success(const_cast<LoonFFIResult*>(result))) {
    return;
  }

  const char* message = loon_ffi_get_errmsg(const_cast<LoonFFIResult*>(result));
  const char* exception_class = "java/lang/RuntimeException";

  // Code 7 is System at the native boundary because it normally describes
  // deployment properties. The Java binding's property map, however, is
  // supplied directly by this API's caller, so expose the language's normal
  // invalid-argument exception without changing the shared native category.
  if (result->err_code == loon_errcode_user_invalid_argument || result->err_code == loon_errcode_invalid_properties) {
    exception_class = "java/lang/IllegalArgumentException";
  } else {
    exception_class = "java/lang/RuntimeException";
  }

  ThrowJavaException(env, exception_class, message);
}

jobjectArray ConvertToJavaStringArray(JNIEnv* env, const char* const* strings, size_t count) {
  if (count > 0 && strings == nullptr) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Native string array is null while count is non-zero");
    return nullptr;
  }

  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    return nullptr;
  }
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(count), string_class, nullptr);
  if (result == nullptr) {
    env->DeleteLocalRef(string_class);
    return nullptr;
  }

  for (size_t i = 0; i < count; ++i) {
    if (strings[i] == nullptr) {
      ThrowJavaException(env, "java/lang/RuntimeException", "Native string array contains a null element");
      env->DeleteLocalRef(string_class);
      return nullptr;
    }
    jstring str = env->NewStringUTF(strings[i]);
    if (str == nullptr) {
      env->DeleteLocalRef(string_class);
      return nullptr;
    }
    env->SetObjectArrayElement(result, static_cast<jsize>(i), str);
    env->DeleteLocalRef(str);
    if (env->ExceptionCheck()) {
      env->DeleteLocalRef(string_class);
      return nullptr;
    }
  }

  env->DeleteLocalRef(string_class);
  return result;
}

const char** ConvertFromJavaStringArray(JNIEnv* env, jobjectArray java_array, size_t* out_count) {
  if (out_count == nullptr) {
    ThrowJavaException(env, "java/lang/IllegalArgumentException", "out_count must not be null");
    return nullptr;
  }

  *out_count = 0;
  if (java_array == nullptr) {
    return nullptr;
  }

  jsize length = env->GetArrayLength(java_array);
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  if (length == 0) {
    return nullptr;
  }

  // calloc both detects size multiplication overflow and makes partial cleanup
  // safe when a later JNI or strdup allocation fails.
  const char** strings = static_cast<const char**>(calloc(static_cast<size_t>(length), sizeof(char*)));
  if (strings == nullptr) {
    ThrowJavaException(env, "java/lang/RuntimeException", "Unexpected native allocation failure for string array");
    return nullptr;
  }

  for (jsize i = 0; i < length; ++i) {
    jstring jstr = static_cast<jstring>(env->GetObjectArrayElement(java_array, i));
    if (env->ExceptionCheck()) {
      FreeStringArray(env, strings, static_cast<size_t>(length));
      return nullptr;
    }
    if (jstr == nullptr) {
      FreeStringArray(env, strings, static_cast<size_t>(length));
      char message[96];
      std::snprintf(message, sizeof(message), "String array element at index %d must not be null", i);
      ThrowJavaException(env, "java/lang/IllegalArgumentException", message);
      return nullptr;
    }

    const char* str = env->GetStringUTFChars(jstr, nullptr);
    if (str == nullptr) {
      env->DeleteLocalRef(jstr);
      FreeStringArray(env, strings, static_cast<size_t>(length));
      // GetStringUTFChars reports allocation failure with a pending OOME.
      return nullptr;
    }

    char* copy = strdup(str);
    env->ReleaseStringUTFChars(jstr, str);
    env->DeleteLocalRef(jstr);
    if (copy == nullptr) {
      FreeStringArray(env, strings, static_cast<size_t>(length));
      ThrowJavaException(env, "java/lang/RuntimeException",
                         "Unexpected native allocation failure while copying string array element");
      return nullptr;
    }
    strings[i] = copy;
  }

  *out_count = static_cast<size_t>(length);
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
