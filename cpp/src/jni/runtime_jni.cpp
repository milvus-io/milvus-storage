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

// Process-wide storage runtime settings a JVM host sizes after start-up.

#include <arrow/io/type_fwd.h>

#include <string>

#include "jni_raii.h"
#include "milvus-storage/thread_pool.h"

using namespace milvus_storage::jni;

extern "C" {
JNIEXPORT jint JNICALL Java_io_milvus_storage_MilvusStorageRuntimeNative_ioThreadPoolCapacity(JNIEnv*, jobject) {
  return static_cast<jint>(arrow::io::GetIOThreadPoolCapacity());
}

JNIEXPORT void JNICALL Java_io_milvus_storage_MilvusStorageRuntimeNative_setArrowIoThreadPoolCapacity(JNIEnv* env,
                                                                                                      jobject,
                                                                                                      jint threads) {
  if (threads <= 0) {
    Throw(env, "java/lang/IllegalArgumentException", "IO thread pool capacity must be positive");
    return;
  }
  auto status = milvus_storage::SetArrowIOThreadPoolCapacity(static_cast<uint32_t>(threads));
  if (!status.ok()) {
    const std::string message = status.ToString();
    Throw(env, "java/lang/IllegalStateException", message.c_str());
  }
}
}  // extern "C"
