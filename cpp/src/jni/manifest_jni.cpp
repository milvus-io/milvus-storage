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
#include <memory>
#include <string>
#include <iostream>

// ==================== JNI Manifest Implementation ====================

extern "C" {

JNIEXPORT jstring JNICALL Java_io_milvus_storage_MilvusStorageManifest_00024_getLatestColumnGroupsNative(
    JNIEnv* env, jobject obj, jstring base_path, jlong properties_ptr) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    char* column_groups = nullptr;
    FFIResult result = get_latest_column_groups(base_path_cstr, properties, &column_groups, nullptr /* read_version */);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return nullptr;
    }

    jstring java_column_groups = env->NewStringUTF(column_groups);
    free_cstr(column_groups);

    return java_column_groups;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get latest column groups: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

// ==================== JNI Transaction Implementation ====================

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionBegin(JNIEnv* env,
                                                                                         jobject obj,
                                                                                         jstring base_path,
                                                                                         jlong properties_ptr) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    TransactionHandle transaction_handle;
    FFIResult result = transaction_begin(base_path_cstr, properties, &transaction_handle, -1 /* read_version */);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return -1;
    }

    return static_cast<jlong>(transaction_handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to begin transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jstring JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionGetColumnGroups(
    JNIEnv* env, jobject obj, jlong transaction_handle) {
  try {
    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);

    char* column_groups = nullptr;
    FFIResult result = transaction_get_column_groups(handle, &column_groups);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return nullptr;
    }

    jstring java_column_groups = env->NewStringUTF(column_groups);
    free_cstr(column_groups);

    return java_column_groups;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get column groups from transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

JNIEXPORT jboolean JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionCommit(
    JNIEnv* env, jobject obj, jlong transaction_handle, jint update_id, jint resolve_id, jstring column_groups) {
  try {
    if (!column_groups) {
      jclass exc_class = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exc_class, "column_groups must not be null");
      return JNI_FALSE;
    }

    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);
    const char* column_groups_cstr = env->GetStringUTFChars(column_groups, nullptr);

    TransactionCommitResult commit_result;
    FFIResult result = transaction_commit(handle, static_cast<int16_t>(update_id), static_cast<int16_t>(resolve_id),
                                          const_cast<char*>(column_groups_cstr), &commit_result);

    env->ReleaseStringUTFChars(column_groups, column_groups_cstr);

    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return JNI_FALSE;
    }

    if (!commit_result.success) {
      std::cerr << "Transaction commit failed: " << commit_result.failed_message << std::endl;
      free_cstr(commit_result.failed_message);
    }

    return commit_result.success ? JNI_TRUE : JNI_FALSE;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to commit transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return JNI_FALSE;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAbort(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong transaction_handle) {
  try {
    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);

    FFIResult result = transaction_abort(handle);
    if (!IsSuccess(&result)) {
      FreeFFIResult(&result);
      ThrowJavaExceptionFromFFIResult(env, &result);
      return;
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to abort transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionDestroy(JNIEnv* env,
                                                                                          jobject obj,
                                                                                          jlong transaction_handle) {
  try {
    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);
    transaction_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

}  // extern "C"
