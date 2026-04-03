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

#include "milvus-storage/common/log.h"

#include <cstdint>
#include <pthread.h>

std::string GetThreadName() {
  char buf[16] = {};
  if (pthread_getname_np(pthread_self(), buf, sizeof(buf)) == 0 && buf[0] != '\0') {
    return buf;
  }
#ifdef __APPLE__
  uint64_t tid;
  pthread_threadid_np(nullptr, &tid);
#else
  auto tid = gettid();
#endif
  return fmt::format("thread-{}", tid);
}
