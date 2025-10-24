// Copyright 2024 Zilliz
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
#include <arrow/io/interfaces.h>
#include <arrow/status.h>
#include <arrow/util/thread_pool.h>
#include <stdexcept>
#include "arrow/util/logging.h"

namespace milvus_storage {

template <typename... SubmitArgs>
auto SubmitIO(arrow::io::IOContext io_context, SubmitArgs&&... submit_args)
    -> decltype(std::declval<::arrow::internal::Executor*>()->Submit(submit_args...)) {
  arrow::internal::TaskHints hints;
  hints.external_id = io_context.external_id();
  return io_context.executor()->Submit(hints, io_context.stop_token(), std::forward<SubmitArgs>(submit_args)...);
};

inline void CloseFromDestructor(arrow::io::FileInterface* file) {
  arrow::Status st = file->Close();
  if (!st.ok()) {
    auto file_type = typeid(*file).name();
    std::stringstream ss;
    ss << "When destroying file of type " << file_type << ": " << st.message();
    ARROW_LOG(ERROR) << st.WithMessage(ss.str());
    throw std::runtime_error(ss.str());
  }
}

}  // namespace milvus_storage