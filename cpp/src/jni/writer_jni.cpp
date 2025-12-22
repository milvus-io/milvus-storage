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
#include <arrow/c/abi.h>
#include <cassert>
#include <memory>
#include <string>

// ==================== JNI Writer Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerNew(
    JNIEnv* env, jobject obj, jstring base_path, jlong schema_ptr, jlong properties_ptr) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    ArrowSchema* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    WriterHandle writer_handle;
    FFIResult result = writer_new(base_path_cstr, schema, properties, &writer_handle);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return -1;
    }

    return static_cast<jlong>(writer_handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to create writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageWriter_writerWrite(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong writer_handle,
                                                                              jlong array_ptr) {
  try {
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);
    ArrowArray* array = reinterpret_cast<ArrowArray*>(array_ptr);

    FFIResult result = writer_write(handle, array);
    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
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
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);

    FFIResult result = writer_flush(handle);
    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
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
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);

    ColumnGroupsHandle column_groups = 0;
    // no need use the metadata parameters
    FFIResult result = writer_close(handle, nullptr, nullptr, 0, &column_groups);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return -1;
    }

    return static_cast<jlong>(column_groups);
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
    WriterHandle handle = static_cast<WriterHandle>(writer_handle);
    writer_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy writer: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}
