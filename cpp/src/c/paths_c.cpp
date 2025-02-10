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

#include "milvus-storage/c/paths_c.h"

#include <vector>
#include <string>
#include <memory>

using Vec = std::vector<std::string>;

CPaths NewCPaths() {
  auto v = std::make_unique<Vec>();
  return v.release();
}

void AddPathToCPaths(CPaths paths, const char* path, uint32_t path_size) {
  auto v = static_cast<Vec*>(paths);
  v->emplace_back(std::string(path, path_size));
}

void FreeCPaths(CPaths paths) { delete static_cast<Vec*>(paths); }