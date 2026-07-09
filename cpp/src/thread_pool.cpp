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

#include "milvus-storage/thread_pool.h"

#include <arrow/io/type_fwd.h>
#include <arrow/util/thread_pool.h>

#include "rust_runtime.h"

namespace milvus_storage {

arrow::Status ConfigureStorageRuntime(uint32_t num_of_cpu_threads, uint32_t num_of_io_threads) {
  if (auto error = ConfigureRustRuntime(num_of_cpu_threads, num_of_io_threads); error.has_value()) {
    return arrow::Status::Invalid("Failed to configure Rust runtime: ", *error);
  }
  if (auto status = arrow::SetCpuThreadPoolCapacity(num_of_cpu_threads); !status.ok()) {
    return arrow::Status::IOError("Failed to configure Arrow CPU thread pool: ", status.ToString());
  }
  if (auto status = arrow::io::SetIOThreadPoolCapacity(num_of_io_threads); !status.ok()) {
    return arrow::Status::IOError("Failed to configure Arrow IO thread pool: ", status.ToString());
  }
  return arrow::Status::OK();
}

}  // namespace milvus_storage
