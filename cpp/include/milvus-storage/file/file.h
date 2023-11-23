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

#include <cstdint>
#include <string>

namespace milvus_storage {
enum class FileType {
  kUnknown,
  kVector,
  kScalar,
  kDelete,
};

class File {
  public:
  File(std::string& file_path, FileType file_type);

  bool is_vector() const;

  bool is_scalar() const;

  bool is_delete() const;

  std::string& path();

  private:
  std::string file_path_;
  FileType file_type_;
};

}  // namespace milvus_storage