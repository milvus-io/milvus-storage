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

#pragma once

#include "milvus-storage/ffi_jni.h"
#include <cstring>
#include <exception>
#include <stdexcept>
#include <new>
#include <utility>
#include <string>
#include <vector>

namespace milvus_storage::jni {

inline void Throw(JNIEnv* env, const char* type, const char* message) {
  if (env->ExceptionCheck())
    return;
  jclass cls = env->FindClass(type);
  if (cls != nullptr) {
    env->ThrowNew(cls, message);
    env->DeleteLocalRef(cls);
  }
}

inline bool CheckResult(JNIEnv* env, LoonFFIResult result) {
  const bool success = loon_ffi_is_success(&result);
  if (!success && !env->ExceptionCheck())
    ThrowJavaExceptionFromFFIResult(env, &result);
  loon_ffi_free_result(&result);
  return success && !env->ExceptionCheck();
}

template <typename T>
class LocalRef {
  public:
  LocalRef(JNIEnv* env, T value) : env_(env), value_(value) {}
  ~LocalRef() {
    if (value_ != nullptr)
      env_->DeleteLocalRef(value_);
  }
  LocalRef(const LocalRef&) = delete;
  LocalRef& operator=(const LocalRef&) = delete;
  T get() const { return value_; }
  T release() {
    T result = value_;
    value_ = nullptr;
    return result;
  }

  private:
  JNIEnv* env_;
  T value_;
};

class ScopedUtf8 {
  public:
  ScopedUtf8(JNIEnv* env, jstring value) : env_(env), value_(value), chars_(nullptr) {
    if (env_->ExceptionCheck())
      return;
    if (value_ == nullptr) {
      Throw(env_, "java/lang/IllegalArgumentException", "String argument must not be null");
      return;
    }
    chars_ = env_->GetStringUTFChars(value_, nullptr);
  }
  ~ScopedUtf8() {
    if (chars_ != nullptr)
      env_->ReleaseStringUTFChars(value_, chars_);
  }
  ScopedUtf8(const ScopedUtf8&) = delete;
  ScopedUtf8& operator=(const ScopedUtf8&) = delete;
  bool valid() const { return chars_ != nullptr; }
  const char* get() const { return chars_; }
  uint32_t size() const { return static_cast<uint32_t>(std::strlen(chars_)); }

  private:
  JNIEnv* env_;
  jstring value_;
  const char* chars_;
};

inline bool ReadStrings(JNIEnv* env, jobjectArray array, std::vector<std::string>* values) {
  if (env->ExceptionCheck())
    return false;
  if (array == nullptr) {
    Throw(env, "java/lang/IllegalArgumentException", "String array must not be null");
    return false;
  }
  const jsize count = env->GetArrayLength(array);
  values->reserve(count);
  for (jsize i = 0; i < count; ++i) {
    LocalRef<jstring> value(env, static_cast<jstring>(env->GetObjectArrayElement(array, i)));
    if (env->ExceptionCheck())
      return false;
    ScopedUtf8 utf8(env, value.get());
    if (!utf8.valid())
      return false;
    values->emplace_back(utf8.get());
  }
  return true;
}

inline std::vector<const char*> StringPointers(const std::vector<std::string>& values) {
  std::vector<const char*> pointers;
  pointers.reserve(values.size());
  for (const auto& value : values) pointers.push_back(value.c_str());
  return pointers;
}

template <typename T, typename F>
T Guard(JNIEnv* env, T failure, F&& body) noexcept {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    Throw(env, "java/lang/OutOfMemoryError", "Native allocation failed");
  } catch (const std::invalid_argument& error) {
    Throw(env, "java/lang/IllegalArgumentException", error.what());
  } catch (const std::exception& error) {
    Throw(env, "java/lang/RuntimeException", error.what());
  } catch (...) {
    Throw(env, "java/lang/RuntimeException", "Unknown native error");
  }
  return failure;
}

template <typename F>
void GuardVoid(JNIEnv* env, F&& body) noexcept {
  Guard<int>(env, 0, [&] {
    body();
    return 0;
  });
}

}  // namespace milvus_storage::jni
