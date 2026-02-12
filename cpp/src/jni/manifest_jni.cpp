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
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    // Begin a transaction to get the latest manifest
    LoonTransactionHandle transaction_handle;
    LoonFFIResult result = loon_transaction_begin(base_path_cstr, properties, -1 /* read_version */,
                                                  1 /* retry_limit */, &transaction_handle);

    if (!loon_ffi_is_success(&result)) {
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Get read version from transaction
    int64_t read_version = 0;
    result = loon_transaction_get_read_version(transaction_handle, &read_version);
    if (!loon_ffi_is_success(&result)) {
      loon_transaction_destroy(transaction_handle);
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Get manifest from transaction
    LoonManifest* manifest = nullptr;
    result = loon_transaction_get_manifest(transaction_handle, &manifest);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!loon_ffi_is_success(&result)) {
      loon_transaction_destroy(transaction_handle);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Return [manifestPtr, readVersion]
    // Note: We return manifest pointer, caller must manage its lifecycle
    jlongArray ret = env->NewLongArray(2);
    jlong values[2] = {reinterpret_cast<jlong>(manifest), read_version};
    env->SetLongArrayRegion(ret, 0, 2, values);

    // Destroy the transaction (manifest is still valid)
    loon_transaction_destroy(transaction_handle);

    return ret;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get latest column groups: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return nullptr;
  }
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getColumnGroupsWithVersion(
    JNIEnv* env, jobject obj, jstring base_path, jlong properties_ptr, jlong read_version) {
  try {
    const char* base_path_cstr = env->GetStringUTFChars(base_path, nullptr);
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    // Begin a transaction with the specified read version
    LoonTransactionHandle transaction_handle;
    LoonFFIResult result =
        loon_transaction_begin(base_path_cstr, properties, read_version, 1 /* retry_limit */, &transaction_handle);

    if (!loon_ffi_is_success(&result)) {
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Get read version from transaction (may differ from requested if requested was -1)
    int64_t actual_read_version = 0;
    result = loon_transaction_get_read_version(transaction_handle, &actual_read_version);
    if (!loon_ffi_is_success(&result)) {
      loon_transaction_destroy(transaction_handle);
      env->ReleaseStringUTFChars(base_path, base_path_cstr);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Get manifest from transaction
    LoonManifest* manifest = nullptr;
    result = loon_transaction_get_manifest(transaction_handle, &manifest);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!loon_ffi_is_success(&result)) {
      loon_transaction_destroy(transaction_handle);
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return nullptr;
    }

    // Return [manifestPtr, actualReadVersion]
    jlongArray ret = env->NewLongArray(2);
    jlong values[2] = {reinterpret_cast<jlong>(manifest), actual_read_version};
    env->SetLongArrayRegion(ret, 0, 2, values);

    // Destroy the transaction (manifest is still valid)
    loon_transaction_destroy(transaction_handle);

    return ret;
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to get column groups with version: " + std::string(e.what());
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
    LoonProperties* properties = reinterpret_cast<LoonProperties*>(properties_ptr);

    LoonTransactionHandle transaction_handle;
    LoonFFIResult result = loon_transaction_begin(base_path_cstr, properties, -1 /* read_version */,
                                                  1 /* retry_limit */, &transaction_handle);

    env->ReleaseStringUTFChars(base_path, base_path_cstr);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
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
    LoonTransactionHandle handle = static_cast<LoonTransactionHandle>(transaction_handle);

    LoonManifest* manifest = nullptr;
    LoonFFIResult result = loon_transaction_get_manifest(handle, &manifest);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
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

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionCommit(
    JNIEnv* env, jobject obj, jlong transaction_handle, jint update_id, jint resolve_id, jlong column_groups) {
  try {
    LoonTransactionHandle handle = static_cast<LoonTransactionHandle>(transaction_handle);

    // If column_groups is provided, apply updates based on update_id
    if (column_groups) {
      LoonColumnGroups* cgroups = reinterpret_cast<LoonColumnGroups*>(column_groups);

      if (update_id == 1) {
        // ADDFIELD: add each column group as a new field column group
        for (uint32_t i = 0; i < cgroups->num_of_column_groups; i++) {
          LoonFFIResult add_result = loon_transaction_add_column_group(handle, &cgroups->column_group_array[i]);
          if (!loon_ffi_is_success(&add_result)) {
            ThrowJavaExceptionFromFFIResult(env, &add_result);
            loon_ffi_free_result(&add_result);
            return -1;
          }
        }
      } else {
        // ADDFILES (default): append files to existing column groups
        LoonFFIResult append_result = loon_transaction_append_files(handle, cgroups);
        if (!loon_ffi_is_success(&append_result)) {
          ThrowJavaExceptionFromFFIResult(env, &append_result);
          loon_ffi_free_result(&append_result);
          return -1;
        }
      }
    }

    int64_t committed_version = 0;
    LoonFFIResult result = loon_transaction_commit(handle, &committed_version);

    if (!loon_ffi_is_success(&result)) {
      ThrowJavaExceptionFromFFIResult(env, &result);
      loon_ffi_free_result(&result);
      return -1;
    }

    return static_cast<jlong>(committed_version);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to commit transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAbort(JNIEnv* env,
                                                                                        jobject obj,
                                                                                        jlong transaction_handle) {
  try {
    LoonTransactionHandle handle = static_cast<LoonTransactionHandle>(transaction_handle);
    // Abort is simply destroying the transaction without committing
    loon_transaction_destroy(handle);
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
    LoonTransactionHandle handle = static_cast<LoonTransactionHandle>(transaction_handle);
    loon_transaction_destroy(handle);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to destroy transaction: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}
