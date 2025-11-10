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
#include <arrow/c/abi.h>
#include <string>

// ==================== JNI Arrow Resource Management ====================

extern "C" {

JNIEXPORT void JNICALL Java_io_milvus_storage_ArrowUtils_00024_releaseArrowArray(JNIEnv* env,
                                                                                 jobject obj,
                                                                                 jlong array_ptr) {
  try {
    ArrowArray* array = reinterpret_cast<ArrowArray*>(array_ptr);
    if (array != nullptr) {
      if (array->release != nullptr) {
        array->release(array);
      }
      free(array);
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to release arrow array: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_io_milvus_storage_ArrowUtils_00024_releaseArrowStream(JNIEnv* env,
                                                                                  jobject obj,
                                                                                  jlong stream_ptr) {
  try {
    ArrowArrayStream* stream = reinterpret_cast<ArrowArrayStream*>(stream_ptr);
    if (stream != nullptr) {
      if (stream->release != nullptr) {
        stream->release(stream);
      }
      free(stream);
    }
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to release arrow stream: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return;
  }
}

JNIEXPORT jlong JNICALL Java_io_milvus_storage_ArrowUtils_00024_readNextBatch(JNIEnv* env,
                                                                              jobject obj,
                                                                              jlong stream_ptr) {
  try {
    ArrowArrayStream* stream = reinterpret_cast<ArrowArrayStream*>(stream_ptr);
    if (stream == nullptr || stream->get_next == nullptr) {
      jclass exc_class = env->FindClass("java/lang/IllegalArgumentException");
      env->ThrowNew(exc_class, "Invalid ArrowArrayStream pointer");
      return -1;
    }

    // Allocate ArrowArray for the next batch
    ArrowArray* array = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray)));
    if (array == nullptr) {
      jclass exc_class = env->FindClass("java/lang/OutOfMemoryError");
      env->ThrowNew(exc_class, "Failed to allocate ArrowArray");
      return -1;
    }

    // Call get_next to read the next batch
    int result = stream->get_next(stream, array);
    if (result != 0) {
      // Error occurred
      const char* error_msg = stream->get_last_error ? stream->get_last_error(stream) : "Unknown error";
      if (array->release != nullptr) {
        array->release(array);
      }
      free(array);
      jclass exc_class = env->FindClass("java/lang/RuntimeException");
      env->ThrowNew(exc_class, error_msg);
      return -1;
    }

    // Check if we've reached the end of stream (release callback is null)
    if (array->release == nullptr) {
      free(array);
      return 0;  // End of stream
    }

    return reinterpret_cast<jlong>(array);
  } catch (const std::exception& e) {
    jclass exc_class = env->FindClass("java/lang/RuntimeException");
    std::string error_msg = "Failed to read next batch: " + std::string(e.what());
    env->ThrowNew(exc_class, error_msg.c_str());
    return -1;
  }
}

}  // extern "C"
