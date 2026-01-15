// Copyright 2023 Zilliz
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

#include "milvus-storage/ffi_c.h"

#include "milvus-storage/ffi_internal/result.h"
#include "milvus-storage/thread_pool.h"

using namespace milvus_storage;
LoonFFIResult loon_thread_pool_singleton(size_t num_of_thread) {
  if (num_of_thread == 0) {
    RETURN_ERROR(LOON_INVALID_ARGS, "num_of_thread must be greater than 0");
  }

  try {
    ThreadPoolHolder::WithSingleton(num_of_thread);
    RETURN_SUCCESS();
  } catch (std::exception& e) {
    RETURN_EXCEPTION(e.what());
  }

  RETURN_UNREACHABLE();
}

void loon_thread_pool_singleton_release() { ThreadPoolHolder::Release(); }
