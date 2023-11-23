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

#include "file/file.h"

namespace milvus_storage {

File::File(std::string& file_path, FileType file_type) : file_path_(std::move(file_path)), file_type_(file_type) {}

bool File::is_vector() const { return file_type_ == FileType::kVector; }

bool File::is_scalar() const { return file_type_ == FileType::kScalar; }

bool File::is_delete() const { return file_type_ == FileType::kDelete; }

std::string& File::path() { return file_path_; }
}  // namespace milvus_storage