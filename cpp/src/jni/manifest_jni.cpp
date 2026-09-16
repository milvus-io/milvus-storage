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
#include "jni_raii.h"
#include <memory>
#include <vector>

using namespace milvus_storage::jni;

namespace {
using ManifestOwner = std::unique_ptr<LoonManifest, decltype(&loon_manifest_destroy)>;
struct TransactionOwner {
  LoonTransactionHandle handle = 0;
  ~TransactionOwner() {
    if (handle != 0)
      loon_transaction_destroy(handle);
  }
};

jlongArray OpenManifest(JNIEnv* env, jstring path, jlong properties, jlong version) {
  ScopedUtf8 p(env, path);
  if (!p.valid())
    return nullptr;
  TransactionOwner transaction;
  if (!CheckResult(env, loon_transaction_begin(p.get(), reinterpret_cast<LoonProperties*>(properties), version,
                                               LOON_TRANSACTION_RESOLVE_FAIL, 1, &transaction.handle)))
    return nullptr;
  int64_t actual_version = 0;
  if (!CheckResult(env, loon_transaction_get_read_version(transaction.handle, &actual_version)))
    return nullptr;
  LoonManifest* manifest = nullptr;
  const auto result = loon_transaction_get_manifest(transaction.handle, &manifest);
  ManifestOwner owned(manifest, &loon_manifest_destroy);
  if (!CheckResult(env, result))
    return nullptr;
  jlongArray output = env->NewLongArray(3);
  if (output == nullptr)
    return nullptr;
  const jlong values[] = {reinterpret_cast<jlong>(manifest), actual_version,
                          reinterpret_cast<jlong>(&manifest->column_groups)};
  env->SetLongArrayRegion(output, 0, 3, values);
  if (env->ExceptionCheck())
    return nullptr;
  owned.release();
  return output;
}
}  // namespace

extern "C" {
JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getLatestColumnGroups(
    JNIEnv* env, jobject, jstring path, jlong properties) {
  return Guard<jlongArray>(env, nullptr, [&] { return OpenManifest(env, path, properties, -1); });
}

JNIEXPORT jlongArray JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_getColumnGroupsWithVersion(
    JNIEnv* env, jobject, jstring path, jlong properties, jlong version) {
  return Guard<jlongArray>(env, nullptr, [&] { return OpenManifest(env, path, properties, version); });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_destroyManifest(JNIEnv* env,
                                                                                          jobject,
                                                                                          jlong pointer) {
  GuardVoid(env, [&] { loon_manifest_destroy(reinterpret_cast<LoonManifest*>(pointer)); });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageManifestNative_columnGroups(JNIEnv* env,
                                                                                        jobject,
                                                                                        jlong pointer) {
  if (pointer == 0) {
    Throw(env, "java/lang/IllegalArgumentException", "Manifest must not be null");
    return 0;
  }
  return reinterpret_cast<jlong>(&reinterpret_cast<LoonManifest*>(pointer)->column_groups);
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionBegin(
    JNIEnv* env, jobject, jstring path, jlong properties, jlong version, jint resolve, jint retries) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return jlong{0};
    if (retries < 0) {
      Throw(env, "java/lang/IllegalArgumentException", "retryLimit must be nonnegative");
      return jlong{0};
    }
    LoonTransactionHandle handle = 0;
    if (!CheckResult(env, loon_transaction_begin(p.get(), reinterpret_cast<LoonProperties*>(properties), version,
                                                 resolve, static_cast<uint32_t>(retries), &handle)))
      return jlong{0};
    return static_cast<jlong>(handle);
  });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionGetColumnGroups(JNIEnv* env,
                                                                                                   jobject,
                                                                                                   jlong handle) {
  return Guard<jlong>(env, 0, [&] {
    LoonManifest* manifest = nullptr;
    const auto result = loon_transaction_get_manifest(static_cast<LoonTransactionHandle>(handle), &manifest);
    ManifestOwner owned(manifest, &loon_manifest_destroy);
    if (!CheckResult(env, result))
      return jlong{0};
    return reinterpret_cast<jlong>(owned.release());
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAppendFiles(JNIEnv* env,
                                                                                              jobject,
                                                                                              jlong handle,
                                                                                              jlong groups) {
  GuardVoid(env, [&] {
    CheckResult(env, loon_transaction_append_files(static_cast<LoonTransactionHandle>(handle),
                                                   reinterpret_cast<const LoonColumnGroups*>(groups)));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAddColumnGroups(JNIEnv* env,
                                                                                                  jobject,
                                                                                                  jlong handle,
                                                                                                  jlong pointer) {
  GuardVoid(env, [&] {
    if (pointer == 0) {
      Throw(env, "java/lang/IllegalArgumentException", "Column groups must not be null");
      return;
    }
    const auto* groups = reinterpret_cast<const LoonColumnGroups*>(pointer);
    for (uint32_t i = 0; i < groups->num_of_column_groups; ++i) {
      if (!CheckResult(env, loon_transaction_add_column_group(static_cast<LoonTransactionHandle>(handle),
                                                              &groups->column_group_array[i])))
        return;
    }
  });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionCommit(JNIEnv* env,
                                                                                          jobject,
                                                                                          jlong handle) {
  return Guard<jlong>(env, -1, [&] {
    int64_t version = 0;
    if (!CheckResult(env, loon_transaction_commit(static_cast<LoonTransactionHandle>(handle), &version)))
      return jlong{-1};
    return static_cast<jlong>(version);
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionDropColumn(JNIEnv* env,
                                                                                             jobject,
                                                                                             jlong handle,
                                                                                             jstring column) {
  GuardVoid(env, [&] {
    ScopedUtf8 name(env, column);
    if (name.valid())
      CheckResult(env, loon_transaction_drop_column(static_cast<LoonTransactionHandle>(handle), name.get()));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionUpdateStat(
    JNIEnv* env, jobject, jlong handle, jstring key, jobjectArray files, jobjectArray keys, jobjectArray values) {
  GuardVoid(env, [&] {
    ScopedUtf8 name(env, key);
    if (!name.valid())
      return;
    std::vector<std::string> file_values, key_values, metadata_values;
    if (!ReadStrings(env, files, &file_values) || !ReadStrings(env, keys, &key_values) ||
        !ReadStrings(env, values, &metadata_values))
      return;
    if (key_values.size() != metadata_values.size()) {
      Throw(env, "java/lang/IllegalArgumentException", "Metadata keys and values must have equal lengths");
      return;
    }
    const auto file_pointers = StringPointers(file_values);
    const auto key_pointers = StringPointers(key_values);
    const auto value_pointers = StringPointers(metadata_values);
    CheckResult(env, loon_transaction_update_stat(static_cast<LoonTransactionHandle>(handle), name.get(),
                                                  file_pointers.data(), file_pointers.size(), key_pointers.data(),
                                                  value_pointers.data(), key_pointers.size()));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAddDeltaLog(
    JNIEnv* env, jobject, jlong handle, jstring path, jlong entries) {
  GuardVoid(env, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return;
    if (entries < 0) {
      Throw(env, "java/lang/IllegalArgumentException", "Delta entry count must be nonnegative");
      return;
    }
    CheckResult(env, loon_transaction_add_delta_log(static_cast<LoonTransactionHandle>(handle), p.get(), entries));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionAbort(JNIEnv* env,
                                                                                        jobject,
                                                                                        jlong handle) {
  GuardVoid(env, [&] { loon_transaction_destroy(static_cast<LoonTransactionHandle>(handle)); });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageTransaction_transactionDestroy(JNIEnv* env,
                                                                                          jobject,
                                                                                          jlong handle) {
  GuardVoid(env, [&] { loon_transaction_destroy(static_cast<LoonTransactionHandle>(handle)); });
}
}  // extern "C"
