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
#include <string>
#include <vector>

// ==================== JNI Properties Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageProperties_allocateProperties(JNIEnv* env, jobject obj) {
  try {
    LoonProperties* properties = static_cast<LoonProperties*>(malloc(sizeof(LoonProperties)));
    if (properties == nullptr) {
      jclass exc_class = env->FindClass("java/lang/OutOfMemoryError");
      env->ThrowNew(exc_class, "Failed to allocate memory for Properties");
      return -1;
    }

    // Initialize the properties structure
    properties->properties = nullptr;
    properties->count = 0;

    return reinterpret_cast<jlong>(properties);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to allocate properties: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_createProperties(JNIEnv* env,
                                                                                       jobject obj,
                                                                                       jobject java_map,
                                                                                       jlong properties_ptr) {
  try {
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    jclass map_class = env->GetObjectClass(java_map);
    jmethodID entry_set_method = env->GetMethodID(map_class, "entrySet", "()Ljava/util/Set;");
    jobject entry_set = env->CallObjectMethod(java_map, entry_set_method);

    jclass set_class = env->GetObjectClass(entry_set);
    jmethodID to_array_method = env->GetMethodID(set_class, "toArray", "()[Ljava/lang/Object;");
    jobjectArray entries = static_cast<jobjectArray>(env->CallObjectMethod(entry_set, to_array_method));

    jsize num_entries = env->GetArrayLength(entries);

    std::vector<const char*> keys, values;
    std::vector<std::string> key_storage, value_storage;

    for (jsize i = 0; i < num_entries; ++i) {
      jobject entry = env->GetObjectArrayElement(entries, i);
      jclass entry_class = env->GetObjectClass(entry);

      jmethodID get_key_method = env->GetMethodID(entry_class, "getKey", "()Ljava/lang/Object;");
      jmethodID get_value_method = env->GetMethodID(entry_class, "getValue", "()Ljava/lang/Object;");

      jstring key_jstr = static_cast<jstring>(env->CallObjectMethod(entry, get_key_method));
      jstring value_jstr = static_cast<jstring>(env->CallObjectMethod(entry, get_value_method));

      const char* key_cstr = env->GetStringUTFChars(key_jstr, nullptr);
      const char* value_cstr = env->GetStringUTFChars(value_jstr, nullptr);

      key_storage.emplace_back(key_cstr);
      value_storage.emplace_back(value_cstr);

      env->ReleaseStringUTFChars(key_jstr, key_cstr);
      env->ReleaseStringUTFChars(value_jstr, value_cstr);
      env->DeleteLocalRef(key_jstr);
      env->DeleteLocalRef(value_jstr);
      env->DeleteLocalRef(entry);
    }

    for (size_t i = 0; i < key_storage.size(); ++i) {
      keys.push_back(key_storage[i].c_str());
      values.push_back(value_storage[i].c_str());
    }

    LoonFFIResult result = loon_properties_create(keys.data(), values.data(), keys.size(), properties);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return;
    }

    env->DeleteLocalRef(entries);
    env->DeleteLocalRef(entry_set);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to create properties: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageProperties_freeProperties(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jlong properties_ptr) {
  LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);
  if (properties != nullptr) {
    loon_properties_free(properties);
  }
  return;
}
