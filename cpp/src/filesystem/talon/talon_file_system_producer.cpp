// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus-storage/filesystem/talon/talon_file_system_producer.h"

#include <string>
#include <utility>

#include "milvus-storage/filesystem/talon/talon_file_system.h"

namespace milvus_storage {

arrow::Result<ArrowFileSystemPtr> TalonFileSystemProducer::Make() {
  if (origin_fs_ == nullptr) {
    return arrow::Status::Invalid("Talon requires a remote provider filesystem");
  }

  std::string bucket = config_.bucket_name;
  while (!bucket.empty() && bucket.back() == '/') {
    bucket.pop_back();
  }
  if (bucket.empty()) {
    return arrow::Status::Invalid("Talon requires a non-empty remote bucket");
  }

  return talon::internal::MakeTalonFileSystem(config_, origin_fs_, std::move(bucket));
}

}  // namespace milvus_storage
