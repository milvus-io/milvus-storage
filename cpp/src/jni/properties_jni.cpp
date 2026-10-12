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
#include <cstdlib>
#include <string>
#include <vector>

using namespace milvus_storage::jni;

namespace {

bool CollectProperties(JNIEnv* env, jobject map, std::vector<std::string>* keys, std::vector<std::string>* values) {
  if (env->ExceptionCheck())
    return false;
  if (map == nullptr) {
    Throw(env, "java/lang/IllegalArgumentException", "Property map must not be null");
    return false;
  }
  LocalRef<jclass> map_class(env, env->GetObjectClass(map));
  if (map_class.get() == nullptr)
    return false;
  jmethodID entry_set = env->GetMethodID(map_class.get(), "entrySet", "()Ljava/util/Set;");
  if (entry_set == nullptr)
    return false;
  LocalRef<jobject> set(env, env->CallObjectMethod(map, entry_set));
  if (env->ExceptionCheck())
    return false;
  if (set.get() == nullptr) {
    Throw(env, "java/lang/IllegalArgumentException", "Property entry set must not be null");
    return false;
  }
  LocalRef<jclass> set_class(env, env->GetObjectClass(set.get()));
  if (set_class.get() == nullptr)
    return false;
  jmethodID to_array = env->GetMethodID(set_class.get(), "toArray", "()[Ljava/lang/Object;");
  if (to_array == nullptr)
    return false;
  LocalRef<jobjectArray> entries(env, static_cast<jobjectArray>(env->CallObjectMethod(set.get(), to_array)));
  if (env->ExceptionCheck())
    return false;
  if (entries.get() == nullptr) {
    Throw(env, "java/lang/IllegalArgumentException", "Property entries must not be null");
    return false;
  }
  LocalRef<jclass> string_class(env, env->FindClass("java/lang/String"));
  if (string_class.get() == nullptr)
    return false;
  jsize count = env->GetArrayLength(entries.get());
  keys->reserve(count);
  values->reserve(count);
  for (jsize i = 0; i < count; ++i) {
    LocalRef<jobject> entry(env, env->GetObjectArrayElement(entries.get(), i));
    if (env->ExceptionCheck())
      return false;
    if (entry.get() == nullptr) {
      Throw(env, "java/lang/IllegalArgumentException", "Property entry must not be null");
      return false;
    }
    LocalRef<jclass> entry_class(env, env->GetObjectClass(entry.get()));
    if (entry_class.get() == nullptr)
      return false;
    jmethodID get_key = env->GetMethodID(entry_class.get(), "getKey", "()Ljava/lang/Object;");
    if (get_key == nullptr)
      return false;
    jmethodID get_value = env->GetMethodID(entry_class.get(), "getValue", "()Ljava/lang/Object;");
    if (get_value == nullptr)
      return false;
    LocalRef<jobject> key(env, env->CallObjectMethod(entry.get(), get_key));
    if (env->ExceptionCheck())
      return false;
    LocalRef<jobject> value(env, env->CallObjectMethod(entry.get(), get_value));
    if (env->ExceptionCheck())
      return false;
    if (key.get() == nullptr || value.get() == nullptr || !env->IsInstanceOf(key.get(), string_class.get()) ||
        !env->IsInstanceOf(value.get(), string_class.get())) {
      Throw(env, "java/lang/IllegalArgumentException", "Property keys and values must be non-null strings");
      return false;
    }
    ScopedUtf8 key_chars(env, static_cast<jstring>(key.get()));
    if (!key_chars.valid())
      return false;
    ScopedUtf8 value_chars(env, static_cast<jstring>(value.get()));
    if (!value_chars.valid())
      return false;
    keys->emplace_back(key_chars.get());
    values->emplace_back(value_chars.get());
  }
  return true;
}

}  // namespace

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageProperties_allocateProperties(JNIEnv* env, jobject) {
  return Guard<jlong>(env, 0, [&]() -> jlong {
    auto* properties = static_cast<LoonProperties*>(calloc(1, sizeof(LoonProperties)));
    if (properties == nullptr) {
      Throw(env, "java/lang/OutOfMemoryError", "Cannot allocate storage properties");
      return 0;
    }
    return reinterpret_cast<jlong>(properties);
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_createProperties(JNIEnv* env,
                                                                                       jobject,
                                                                                       jobject java_map,
                                                                                       jlong properties_ptr) {
  GuardVoid(env, [&] {
    if (properties_ptr == 0) {
      Throw(env, "java/lang/IllegalArgumentException", "Properties handle must not be null");
      return;
    }
    std::vector<std::string> keys;
    std::vector<std::string> values;
    if (!CollectProperties(env, java_map, &keys, &values))
      return;
    std::vector<const char*> key_ptrs;
    std::vector<const char*> value_ptrs;
    key_ptrs.reserve(keys.size());
    value_ptrs.reserve(values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      key_ptrs.push_back(keys[i].c_str());
      value_ptrs.push_back(values[i].c_str());
    }
    LoonProperties replacement{};
    if (!CheckResult(env, loon_properties_create(key_ptrs.data(), value_ptrs.data(), keys.size(), &replacement))) {
      loon_properties_free(&replacement);
      return;
    }
    auto* properties = reinterpret_cast<LoonProperties*>(properties_ptr);
    loon_properties_free(properties);
    *properties = replacement;
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_freeProperties(JNIEnv*,
                                                                                     jobject,
                                                                                     jlong properties_ptr) {
  auto* properties = reinterpret_cast<LoonProperties*>(properties_ptr);
  if (properties != nullptr) {
    loon_properties_free(properties);
    free(properties);
  }
}
