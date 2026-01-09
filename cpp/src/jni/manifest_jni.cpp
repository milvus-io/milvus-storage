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

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getLatestColumnGroups(
    JNIEnv* env, jobject obj, jstring base_path, jlong properties_ptr) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    Properties* properties = reinterpret_cast<Properties*>(properties_ptr);

    // Begin a transaction to get the latest manifest
    TransactionHandle transaction_handle;
    FFIResult result =
        transaction_begin(base_path_cstr, properties, -1 /* read_version */, 1 /* retry_limit */, &transaction_handle);

    if (!IsSuccess(&result)) {
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return nullptr;
    }

    // Get read version from transaction
    int64_t read_version = 0;
    result = transaction_get_read_version(transaction_handle, &read_version);
    if (!IsSuccess(&result)) {
      transaction_destroy(transaction_handle);
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return nullptr;
    }

    // Get manifest from transaction
    CManifest* manifest = nullptr;
    result = transaction_get_manifest(transaction_handle, &manifest);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      transaction_destroy(transaction_handle);
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return nullptr;
    }

    // Return [manifestPtr, readVersion]
    // Note: We return manifest pointer, caller must manage its lifecycle
    jlongArray ret = env->NewLongArray(2);
    jlong values[2] = {reinterpret_cast<jlong>(manifest), read_version};
    env->SetLongArrayRegion(ret, 0, 2, values);

    // Destroy the transaction (manifest is still valid)
    transaction_destroy(transaction_handle);

    return ret;
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
    FFIResult result =
        transaction_begin(base_path_cstr, properties, -1 /* read_version */, 1 /* retry_limit */, &transaction_handle);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
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

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionGetColumnGroups(
    JNIEnv* env, jobject obj, jlong transaction_handle) {
  try {
    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);

    CManifest* manifest = nullptr;
    FFIResult result = transaction_get_manifest(handle, &manifest);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return -1;
    }

    // Return manifest pointer (which contains column_groups)
    return reinterpret_cast<jlong>(manifest);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get column groups from transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT jboolean JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionCommit(
    JNIEnv* env, jobject obj, jlong transaction_handle, jint update_id, jint resolve_id, jlong column_groups) {
  try {
    TransactionHandle handle = static_cast<TransactionHandle>(transaction_handle);

    // If column_groups is provided, append files to the transaction
    if (column_groups) {
      CColumnGroups* cgroups = reinterpret_cast<CColumnGroups*>(column_groups);
      FFIResult append_result = transaction_append_files(handle, cgroups);
      if (!IsSuccess(&append_result)) {
        ThrowJavaExceptionFromFFIResult(env, &append_result);
        FreeFFIResult(&append_result);
        return JNI_FALSE;
      }
    }

    int64_t committed_version = 0;
    FFIResult result = transaction_commit(handle, &committed_version);

    if (!IsSuccess(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      FreeFFIResult(&result);
      return JNI_FALSE;
    }

    return JNI_TRUE;
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
    // Abort is simply destroying the transaction without committing
    transaction_destroy(handle);
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
