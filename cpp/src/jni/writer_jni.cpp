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
#include <arrow/c/abi.h>
#include <cassert>
#include <memory>
#include <string>

using namespace milvus_storage::jni;

// ==================== JNI Writer Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerNew(
    JNIEnv* env, jobject obj, jstring base_path, jlong schema_ptr, jlong properties_ptr) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 path(env, base_path);
    if (!path.valid())
      return jlong{0};
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    LoonWriterHandle writer_handle = 0;
    if (!CheckResult(env, loon_writer_new(path.get(), schema, properties, &writer_handle)))
      return jlong{0};

    return static_cast<jlong>(writer_handle);
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerWrite(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong writer_handle,
                                                                              jlong array_ptr) {
  try {
    LoonWriterHandle handle = static_cast<LoonWriterHandle>(writer_handle);
    ArrowArray* array = reinterpret_cast<ArrowArray*>(array_ptr);

    LoonFFIResult result = loon_writer_write(handle, array);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return;
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to write to writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerFlush(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong writer_handle) {
  try {
    LoonWriterHandle handle = static_cast<LoonWriterHandle>(writer_handle);

    LoonFFIResult result = loon_writer_flush(handle);
    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return;
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to flush writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerClose(JNIEnv* env,
                                                                               jobject obj,
                                                                               jlong writer_handle) {
  try {
    LoonWriterHandle handle = static_cast<LoonWriterHandle>(writer_handle);

    LoonColumnGroups* column_groups = nullptr;
    // no need use the metadata parameters
    LoonFFIResult result = loon_writer_close(handle, nullptr, nullptr, 0, &column_groups);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return reinterpret_cast<jlong>(column_groups);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to close writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerDestroy(JNIEnv* env,
                                                                                jobject obj,
                                                                                jlong writer_handle) {
  try {
    LoonWriterHandle handle = static_cast<LoonWriterHandle>(writer_handle);
    loon_writer_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}
