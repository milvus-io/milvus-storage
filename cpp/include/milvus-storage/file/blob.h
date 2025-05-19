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

#include <string>
#include <vector>
#include "proto/manifest.pb.h"
#include "milvus-storage/common/result.h"

namespace milvus_storage {
struct Blob {
  std::string name;
  int64_t size;
  std::string file;

  [[nodiscard]] std::unique_ptr<manifest_proto::Blob> ToProtobuf() const;
  static Blob FromProtobuf(const manifest_proto::Blob blob);
};

using BlobVector = std::vector<Blob>;
}  // namespace milvus_storage
