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
#include "milvus-storage/packed_writer_c.h"

#include <jni.h>
#include <arrow/c/abi.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" {

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusPackedWriter_writerNew(JNIEnv* env,
                                                                            jobject obj,
                                                                            jobjectArray jpaths,
                                                                            jintArray jgroup_offsets,
                                                                            jintArray jgroup_indices,
                                                                            jlong schema_ptr,
                                                                            jlong properties_ptr,
                                                                            jlong buffer_size) {
  // Owning copies of the path strings — must outlive loon_packed_writer_new.
  std::vector<std::string> owned_paths;
  std::vector<const char*> path_ptrs;
  jsize num_paths = 0;

  try {
    if (jpaths == nullptr || jgroup_offsets == nullptr || jgroup_indices == nullptr) {
      jclass exc_class = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exc_class, "paths/group_offsets/group_indices must not be null");
      return -1;
    }

    num_paths = env->GetArrayLength(jpaths);
    owned_paths.reserve(static_cast<size_t>(num_paths));
    path_ptrs.reserve(static_cast<size_t>(num_paths));
    for (jsize i = 0; i < num_paths; ++i) {
      auto jstr = static_cast<jstring>(env->GetObjectArrayElement(jpaths, i));
      const char* utf = env->GetStringUTFChars(jstr, nullptr);
      owned_paths.emplace_back(utf);
      env->ReleaseStringUTFChars(jstr, utf);
      env->DeleteLocalRef(jstr);
      path_ptrs.push_back(owned_paths.back().c_str());
    }

    jsize num_offsets = env->GetArrayLength(jgroup_offsets);
    jsize num_indices = env->GetArrayLength(jgroup_indices);

    if (num_offsets != num_paths + 1) {
      jclass exc_class = env->FindClass("java/lang/IllegalArgumentException");
      std::string msg = "group_offsets length must equal num_paths + 1, got " + std::to_string(num_offsets) + " vs " +
                        std::to_string(num_paths + 1);
      env->ThrowNew(exc_class, msg.c_str());
      return -1;
    }

    // GetIntArrayElements returns jint* aliasing or copying — either way, we
    // must release it once we're done. Since loon_packed_writer_new copies the
    // data into its own vectors before returning, we can release immediately
    // after the call.
    jint* offsets_ptr = env->GetIntArrayElements(jgroup_offsets, nullptr);
    jint* indices_ptr = env->GetIntArrayElements(jgroup_indices, nullptr);

    LoonPackedWriterHandle handle = 0;
    LoonFFIResult result = loon_packed_writer_new(
        path_ptrs.data(), static_cast<int32_t>(num_paths), reinterpret_cast<const int32_t*>(offsets_ptr),
        reinterpret_cast<const int32_t*>(indices_ptr), static_cast<int32_t>(num_indices),
        reinterpret_cast<ArrowSchema*>(schema_ptr), reinterpret_cast<LoonProperties*>(properties_ptr),
        static_cast<int64_t>(buffer_size), &handle);

    env->ReleaseIntArrayElements(jgroup_offsets, offsets_ptr, JNI_ABORT);
    env->ReleaseIntArrayElements(jgroup_indices, indices_ptr, JNI_ABORT);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }
    return static_cast<jlong>(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to create packed writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusPackedWriter_writerWrite(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong writer_handle,
                                                                             jlong array_ptr) {
  try {
    auto handle = static_cast<LoonPackedWriterHandle>(writer_handle);
    auto* array = reinterpret_cast<ArrowArray*>(array_ptr);
    LoonFFIResult result = loon_packed_writer_write(handle, array);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return;
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to write to packed writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusPackedWriter_writerClose(JNIEnv* env,
                                                                             jobject obj,
                                                                             jlong writer_handle) {
  try {
    auto handle = static_cast<LoonPackedWriterHandle>(writer_handle);
    LoonFFIResult result = loon_packed_writer_close(handle);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return;
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to close packed writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusPackedWriter_writerDestroy(JNIEnv* env,
                                                                               jobject obj,
                                                                               jlong writer_handle) {
  try {
    auto handle = static_cast<LoonPackedWriterHandle>(writer_handle);
    loon_packed_writer_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy packed writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

}  // extern "C"
