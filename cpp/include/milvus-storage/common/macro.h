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

#pragma once

#include <parquet/exception.h>

namespace milvus_storage {

#ifndef NDEBUG
// use to measure execution time for debug builds
//
#define DEBUG_TIMER_START(name) auto name##_start = std::chrono::high_resolution_clock::now();
#define DEBUG_TIMER_END(name)                                                                                \
  auto name##_end = std::chrono::high_resolution_clock::now();                                               \
  auto name##_duration = std::chrono::duration_cast<std::chrono::microseconds>(name##_end - name##_start);   \
  std::cout << "Case-" << #name << " Execute time: (" << name##_duration.count() << " us), ("                \
            << name##_duration.count() / 1000.0 << " ms), (" << name##_duration.count() / 1000000.0 << " s)" \
            << std::endl;
#endif

#define CONCAT_IMPL(x, y) x##y

#define CONCAT(x, y) CONCAT_IMPL(x, y)

#undef RETURN_NOT_OK
#define RETURN_NOT_OK(status) \
  do {                        \
    if (!(status).ok()) {     \
      return (status);        \
    }                         \
  } while (false)

#define RETURN_ARROW_NOT_OK(status)                       \
  do {                                                    \
    if (!(status).ok()) {                                 \
      return arrow::Status::Invalid((status).ToString()); \
    }                                                     \
  } while (false)

#define ASSIGN_OR_RETURN_NOT_OK_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                                 \
  RETURN_NOT_OK(status_name.status());                        \
  lhs = std::move(status_name).ValueOrDie();

#define ASSIGN_OR_RETURN_NOT_OK(lhs, rexpr) ASSIGN_OR_RETURN_NOT_OK_IMPL(CONCAT(_tmp_value, __COUNTER__), lhs, rexpr);

#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)
#define UNLIKELY(x)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)
#define UNLIKELY(x)
#endif

#ifndef NDEBUG

#define assert_if(condition, expr) \
  do {                             \
    if (condition) {               \
      assert((expr));              \
    }                              \
  } while (0)

#else
#define assert_if(condition, expr)
#endif

}  // namespace milvus_storage
