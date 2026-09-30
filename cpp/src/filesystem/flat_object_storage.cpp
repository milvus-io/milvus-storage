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

#include "milvus-storage/filesystem/flat_object_storage.h"

#include <string>
#include <utility>
#include <vector>

#include <arrow/filesystem/filesystem.h>

namespace milvus_storage {

using ::arrow::fs::FileInfo;
using ::arrow::fs::FileSelector;
using ::arrow::fs::FileType;

arrow::Result<std::vector<FileInfo>> ListObjectsByPrefix(const std::shared_ptr<arrow::fs::FileSystem>& fs,
                                                         const std::string& prefix) {
  if (auto flat = std::dynamic_pointer_cast<FlatObjectStorage>(fs)) {
    auto result = flat->ListObjectsByPrefix(prefix);
    // NotImplemented is the proxy's "the backend has no native flat listing"
    // signal; anything else (success or a real error) is authoritative.
    if (!result.status().IsNotImplemented()) {
      return result;
    }
  }

  // Fallback for non-object-store backends (local, Azure): Arrow's FileSelector
  // is directory-based, so list the prefix's parent directory recursively and
  // filter file entries on the raw prefix. Bounded and cheap on these backends.
  FileSelector selector;
  auto slash_pos = prefix.find_last_of('/');
  selector.base_dir = slash_pos == std::string::npos ? "" : prefix.substr(0, slash_pos);
  selector.recursive = true;
  selector.allow_not_found = true;

  ARROW_ASSIGN_OR_RAISE(auto infos, fs->GetFileInfo(selector));
  std::vector<FileInfo> result;
  for (auto& info : infos) {
    if (info.type() == FileType::File && info.path().rfind(prefix, 0) == 0) {
      result.push_back(std::move(info));
    }
  }
  return result;
}

arrow::Status DeleteObject(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::string& path) {
  if (auto flat = std::dynamic_pointer_cast<FlatObjectStorage>(fs)) {
    auto status = flat->DeleteObject(path);
    if (!status.IsNotImplemented()) {
      return status;
    }
  }

  // Fallback: DeleteFile, but swallow a positively-confirmed not-found target so
  // removal stays idempotent. Any other DeleteFile error is returned as-is.
  auto status = fs->DeleteFile(path);
  if (status.ok()) {
    return status;
  }
  auto info = fs->GetFileInfo(path);
  if (info.ok() && info->type() == FileType::NotFound) {
    return arrow::Status::OK();
  }
  return status;
}

arrow::Result<bool> ObjectExists(const std::shared_ptr<arrow::fs::FileSystem>& fs, const std::string& path) {
  if (auto flat = std::dynamic_pointer_cast<FlatObjectStorage>(fs)) {
    auto result = flat->ObjectExists(path);
    if (!result.status().IsNotImplemented()) {
      return result;
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto info, fs->GetFileInfo(path));
  return info.type() == FileType::File;
}

}  // namespace milvus_storage
