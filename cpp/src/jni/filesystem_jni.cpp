// Copyright 2026 Zilliz
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

#include "milvus-storage/ffi_filesystem_c.h"
#include "jni_raii.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace milvus_storage::jni;

namespace {
jbyteArray Bytes(JNIEnv* env, const uint8_t* data, uint64_t size) {
  if (size > static_cast<uint64_t>(std::numeric_limits<jsize>::max())) {
    Throw(env, "java/lang/IllegalArgumentException", "File data exceeds the JVM byte array limit");
    return nullptr;
  }
  jbyteArray result = env->NewByteArray(static_cast<jsize>(size));
  if (result != nullptr && size != 0) {
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(size), reinterpret_cast<const jbyte*>(data));
  }
  return env->ExceptionCheck() ? nullptr : result;
}

struct FileList {
  LoonFileInfoList value{nullptr, 0};
  ~FileList() { loon_filesystem_free_file_info_list(&value); }
};
}  // namespace

extern "C" {
JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_create(JNIEnv* env,
                                                                                    jobject,
                                                                                    jlong properties,
                                                                                    jstring path) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return jlong{0};
    FileSystemHandle handle = 0;
    if (!CheckResult(
            env, loon_filesystem_get(reinterpret_cast<const LoonProperties*>(properties), p.get(), p.size(), &handle)))
      return jlong{0};
    return static_cast<jlong>(handle);
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_destroy(JNIEnv* env,
                                                                                    jobject,
                                                                                    jlong handle) {
  GuardVoid(env, [&] { loon_filesystem_destroy(static_cast<FileSystemHandle>(handle)); });
}

JNIEXPORT jbyteArray JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_readFileAll(JNIEnv* env,
                                                                                              jobject,
                                                                                              jlong handle,
                                                                                              jstring path) {
  return Guard<jbyteArray>(env, nullptr, [&]() -> jbyteArray {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return nullptr;
    uint8_t* data = nullptr;
    uint64_t size = 0;
    const auto result =
        loon_filesystem_read_file_all(static_cast<FileSystemHandle>(handle), p.get(), p.size(), &data, &size);
    std::unique_ptr<uint8_t, decltype(&std::free)> owned(data, &std::free);
    if (!CheckResult(env, result))
      return nullptr;
    return Bytes(env, data, size);
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_writeFile(
    JNIEnv* env, jobject, jlong handle, jstring path, jbyteArray data) {
  GuardVoid(env, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return;
    if (data == nullptr) {
      Throw(env, "java/lang/IllegalArgumentException", "data must not be null");
      return;
    }
    const jsize size = env->GetArrayLength(data);
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (size != 0)
      env->GetByteArrayRegion(data, 0, size, reinterpret_cast<jbyte*>(buffer.data()));
    if (env->ExceptionCheck())
      return;
    // The C API requires a non-null data pointer even for an empty file.
    const uint8_t empty = 0;
    CheckResult(env, loon_filesystem_write_file(static_cast<FileSystemHandle>(handle), p.get(), p.size(),
                                                size == 0 ? &empty : buffer.data(), size, nullptr, 0));
  });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_fileSize(JNIEnv* env,
                                                                                      jobject,
                                                                                      jlong handle,
                                                                                      jstring path) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return jlong{0};
    uint64_t size = 0;
    if (!CheckResult(env,
                     loon_filesystem_get_file_info(static_cast<FileSystemHandle>(handle), p.get(), p.size(), &size)))
      return jlong{0};
    if (size > static_cast<uint64_t>(std::numeric_limits<jlong>::max())) {
      Throw(env, "java/lang/ArithmeticException", "File size exceeds Long.MaxValue");
      return jlong{0};
    }
    return static_cast<jlong>(size);
  });
}

JNIEXPORT jboolean JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_exists(JNIEnv* env,
                                                                                       jobject,
                                                                                       jlong handle,
                                                                                       jstring path) {
  return Guard<jboolean>(env, JNI_FALSE, [&]() -> jboolean {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return JNI_FALSE;
    bool exists = false;
    auto result = loon_filesystem_get_path_info(static_cast<FileSystemHandle>(handle), p.get(), p.size(), &exists,
                                                nullptr, nullptr);
    if (result.err_code == loon_errcode_file_not_found) {
      loon_ffi_free_result(&result);
      return JNI_FALSE;
    }
    if (!CheckResult(env, result))
      return JNI_FALSE;
    return exists ? JNI_TRUE : JNI_FALSE;
  });
}

JNIEXPORT jobjectArray JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_list(
    JNIEnv* env, jobject, jlong handle, jstring path, jboolean recursive) {
  return Guard<jobjectArray>(env, nullptr, [&]() -> jobjectArray {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return nullptr;
    FileList list;
    if (!CheckResult(env, loon_filesystem_list_dir(static_cast<FileSystemHandle>(handle), p.get(), p.size(),
                                                   recursive == JNI_TRUE, &list.value)))
      return nullptr;
    if (list.value.count > static_cast<uint32_t>(std::numeric_limits<jsize>::max())) {
      Throw(env, "java/lang/IllegalArgumentException", "Directory listing exceeds the JVM array limit");
      return nullptr;
    }
    LocalRef<jclass> cls(env, env->FindClass("io/milvus/storage/MilvusStorageFileInfo"));
    if (cls.get() == nullptr)
      return nullptr;
    jmethodID ctor = env->GetMethodID(cls.get(), "<init>", "(Ljava/lang/String;ZJJ)V");
    if (ctor == nullptr)
      return nullptr;
    LocalRef<jobjectArray> result(env, env->NewObjectArray(static_cast<jsize>(list.value.count), cls.get(), nullptr));
    if (result.get() == nullptr)
      return nullptr;
    for (uint32_t i = 0; i < list.value.count; ++i) {
      const auto& item = list.value.entries[i];
      const std::string name(item.path == nullptr ? "" : item.path, item.path == nullptr ? 0 : item.path_len);
      LocalRef<jstring> name_ref(env, env->NewStringUTF(name.c_str()));
      if (name_ref.get() == nullptr)
        return nullptr;
      LocalRef<jobject> entry(env, env->NewObject(cls.get(), ctor, name_ref.get(), item.is_dir ? JNI_TRUE : JNI_FALSE,
                                                  static_cast<jlong>(item.size), static_cast<jlong>(item.mtime_ns)));
      if (env->ExceptionCheck() || entry.get() == nullptr)
        return nullptr;
      env->SetObjectArrayElement(result.get(), static_cast<jsize>(i), entry.get());
      if (env->ExceptionCheck())
        return nullptr;
    }
    return result.release();
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_deleteFile(JNIEnv* env,
                                                                                       jobject,
                                                                                       jlong handle,
                                                                                       jstring path) {
  GuardVoid(env, [&] {
    ScopedUtf8 p(env, path);
    if (p.valid())
      CheckResult(env, loon_filesystem_delete_file(static_cast<FileSystemHandle>(handle), p.get(), p.size()));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_createDir(
    JNIEnv* env, jobject, jlong handle, jstring path, jboolean recursive) {
  GuardVoid(env, [&] {
    ScopedUtf8 p(env, path);
    if (p.valid())
      CheckResult(env, loon_filesystem_create_dir(static_cast<FileSystemHandle>(handle), p.get(), p.size(),
                                                  recursive == JNI_TRUE));
  });
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_openReader(
    JNIEnv* env, jobject, jlong handle, jstring path, jlong size) {
  return Guard<jlong>(env, 0, [&] {
    ScopedUtf8 p(env, path);
    if (!p.valid())
      return jlong{0};
    if (size < -1) {
      Throw(env, "java/lang/IllegalArgumentException", "fileSize must be -1 or nonnegative");
      return jlong{0};
    }
    FileSystemReaderHandle reader = 0;
    if (!CheckResult(env, loon_filesystem_open_reader(static_cast<FileSystemHandle>(handle), p.get(), p.size(),
                                                      size < 0 ? 0 : static_cast<uint64_t>(size), &reader)))
      return jlong{0};
    return static_cast<jlong>(reader);
  });
}

JNIEXPORT jbyteArray JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_readerReadAt(
    JNIEnv* env, jobject, jlong handle, jlong offset, jlong length) {
  return Guard<jbyteArray>(env, nullptr, [&]() -> jbyteArray {
    if (offset < 0 || length < 0 || length > std::numeric_limits<jsize>::max() ||
        offset > std::numeric_limits<jlong>::max() - length) {
      Throw(env, "java/lang/IllegalArgumentException", "Invalid file read range");
      return nullptr;
    }
    if (length == 0)
      return Bytes(env, nullptr, 0);
    std::vector<uint8_t> buffer(static_cast<size_t>(length));
    if (!CheckResult(env, loon_filesystem_reader_readat(static_cast<FileSystemReaderHandle>(handle), offset, length,
                                                        buffer.data())))
      return nullptr;
    return Bytes(env, buffer.data(), static_cast<uint64_t>(length));
  });
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageFileSystemNative_readerDestroy(JNIEnv* env,
                                                                                          jobject,
                                                                                          jlong handle) {
  GuardVoid(env, [&] { loon_filesystem_reader_destroy(static_cast<FileSystemReaderHandle>(handle)); });
}
}  // extern "C"
