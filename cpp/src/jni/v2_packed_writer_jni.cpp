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
#include "milvus-storage/ffi_internal/v2_packed_writer_c.h"
#include "jni_raii.h"

#include <jni.h>
#include <arrow/c/abi.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace milvus_storage::jni;

extern "C" {

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusPackedWriter_writerNew(JNIEnv* env,
                                                                            jobject obj,
                                                                            jobjectArray jpaths,
                                                                            jintArray jgroup_offsets,
                                                                            jintArray jgroup_indices,
                                                                            jlong schema_ptr,
                                                                            jlong properties_ptr,
                                                                            jlong buffer_size) {
  return Guard<jlong>(env, 0, [&] {
    if (jpaths == nullptr || jgroup_offsets == nullptr || jgroup_indices == nullptr) {
      Throw(env, "java/lang/IllegalArgumentException", "paths/group_offsets/group_indices must not be null");
      return jlong{0};
    }

    // Own every string and array copy until the C call returns, including when
    // a JVM allocation fails or a C++ allocation throws.
    std::vector<std::string> owned_paths;
    if (!ReadStrings(env, jpaths, &owned_paths))
      return jlong{0};
    const auto path_ptrs = StringPointers(owned_paths);
    const auto num_paths = static_cast<jsize>(owned_paths.size());

    const jsize num_offsets = env->GetArrayLength(jgroup_offsets);
    const jsize num_indices = env->GetArrayLength(jgroup_indices);

    if (static_cast<int64_t>(num_offsets) != static_cast<int64_t>(num_paths) + 1) {
      Throw(env, "java/lang/IllegalArgumentException", "group_offsets length must equal num_paths + 1");
      return jlong{0};
    }

    std::vector<jint> offsets(static_cast<size_t>(num_offsets));
    env->GetIntArrayRegion(jgroup_offsets, 0, num_offsets, offsets.data());
    if (env->ExceptionCheck())
      return jlong{0};
    // Keep a non-null pointer for the C API even when the array is empty.
    std::vector<jint> indices(static_cast<size_t>(num_indices == 0 ? 1 : num_indices));
    if (num_indices != 0)
      env->GetIntArrayRegion(jgroup_indices, 0, num_indices, indices.data());
    if (env->ExceptionCheck())
      return jlong{0};

    LoonPackedWriterHandle handle = 0;
    const LoonFFIResult result = loon_packed_writer_new(
        path_ptrs.data(), static_cast<int32_t>(num_paths), reinterpret_cast<const int32_t*>(offsets.data()),
        reinterpret_cast<const int32_t*>(indices.data()), static_cast<int32_t>(num_indices),
        reinterpret_cast<ArrowSchema*>(schema_ptr), reinterpret_cast<LoonProperties*>(properties_ptr),
        static_cast<int64_t>(buffer_size), &handle);

    if (!CheckResult(env, result))
      return jlong{0};
    return static_cast<jlong>(handle);
  });
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
